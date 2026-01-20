"""
Main orchestrator for Fina chatbot using LangGraph.
Combines patterns from property-sales and bappeda projects.
"""
import asyncio
import json
from typing import Dict, Literal
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.services.llm_service import llm_service
from app.services.vanna_service import vanna_service
from app.services.embedding_service import embedding_service
from app.tools.elasticsearch_tool import elasticsearch_tool
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Router prompt for LLM-based intent classification
ROUTER_PROMPT = """You are an intent classifier for Fina, a fraud detection chatbot assistant.

Classify the user's query into one of FOUR types:

**GREETING (Casual conversation):**
- Greetings, casual chat, general pleasantries
- Keywords: "hello", "hi", "hey", "good morning", "how are you", "what's up", "thanks", "thank you", "bye"
- Examples:
  - "Hello"
  - "Hi there"
  - "How are you?"
  - "Thanks!"

**SQL (Transaction Data - "What happened in our data?"):**
- Factual questions about OUR transaction database
- Keywords: "how many", "which merchants", "fraud rate", "transaction amount", "daily/monthly", "top 10", "statistics", "trend in our data", "show me", "list"
- Examples:
  - "How many fraudulent transactions last month?"
  - "Which category has highest fraud?"
  - "Show top merchants by fraud rate"

**RAG (Theory/Research - "Why/How does it work?"):**
- Conceptual/theoretical questions about fraud detection
- Keywords: "what are", "how does", "methods", "techniques", "according to", "detection system", "components", "cross-border", "report says", "authors recommend", "primary methods"
- Examples:
  - "What are common fraud methods?"
  - "How does fraud detection work?"
  - "What are the components of fraud detection system?"
  - "How much higher are fraud rates for cross-border transactions?"

**HYBRID (Data + Context - "What + Why?"):**
- Questions needing BOTH factual data AND theoretical context
- Patterns: "show data AND explain", "compare our data with standards", "why does [pattern] happen", "analyze and explain"
- Examples:
  - "Which merchants have high fraud and why are they vulnerable?"
  - "Show our fraud trend and explain the pattern"
  - "Analyze our gas_transport fraud and explain if it's normal"

User Query: {query}

Respond with ONLY a JSON object:
{{
    "type": "greeting|sql|rag|hybrid",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this classification"
}}
"""


class FinaOrchestrator:
    """Main orchestrator for Fina chatbot."""

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("greeting_handler", self._handle_greeting)
        workflow.add_node("sql_agent", self._run_sql_agent)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("hybrid_agent", self._run_hybrid_agent)
        workflow.add_node("answer_generation", self._generate_answer)
        workflow.add_node("quality_scoring", self._calculate_quality_score)

        # Set entry point
        workflow.set_entry_point("router")

        # Conditional routing from router
        workflow.add_conditional_edges(
            "router",
            self._decide_route,
            {
                "greeting": "greeting_handler",
                "sql": "sql_agent",
                "rag": "rag_agent",
                "hybrid": "hybrid_agent",
            }
        )

        # Greeting goes directly to quality scoring (skip answer generation)
        workflow.add_edge("greeting_handler", "quality_scoring")

        # All other paths lead to answer generation
        workflow.add_edge("sql_agent", "answer_generation")
        workflow.add_edge("rag_agent", "answer_generation")
        workflow.add_edge("hybrid_agent", "answer_generation")

        # Answer generation leads to quality scoring
        workflow.add_edge("answer_generation", "quality_scoring")

        # Quality scoring is the end
        workflow.add_edge("quality_scoring", END)

        return workflow.compile()

    def _decide_route(self, state: AgentState) -> Literal["greeting", "sql", "rag", "hybrid"]:
        """Decision function for routing."""
        return state["query_type"]

    async def _route_query(self, state: AgentState) -> AgentState:
        """Router node: Classify query using LLM."""
        query = state["user_query"]

        logger.info(f"Routing query: {query}")

        try:
            # Use LLM to classify intent
            prompt = ROUTER_PROMPT.format(query=query)
            result = await llm_service.extract_json(prompt)

            state["query_type"] = result.get("type", "rag")
            state["classification_confidence"] = result.get("confidence", 0.5)
            state["classification_reasoning"] = result.get("reasoning", "")

            logger.info(f"Classified as: {state['query_type']} (confidence: {state['classification_confidence']:.2f})")
            logger.info(f"Reasoning: {state['classification_reasoning']}")

        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Default to RAG on error
            state["query_type"] = "rag"
            state["classification_confidence"] = 0.3
            state["classification_reasoning"] = "Error in classification, defaulting to RAG"

        return state

    async def _handle_greeting(self, state: AgentState) -> AgentState:
        """Handle greeting queries with friendly, concise responses."""
        query = state["user_query"].lower()

        logger.info(f"Handling greeting: {query}")

        # Simple greeting responses
        greetings = {
            "hello": "Hello! I'm Fina, your fraud detection assistant. I can help you analyze fraud patterns in transaction data and answer questions about fraud detection methods. What would you like to know?",
            "hi": "Hi there! I'm Fina. Ask me anything about fraud detection, transaction patterns, or fraud prevention methods.",
            "hey": "Hey! I'm Fina, here to help with fraud detection insights. What can I assist you with today?",
            "good morning": "Good morning! I'm Fina, your fraud detection assistant. How can I help you today?",
            "good afternoon": "Good afternoon! I'm Fina. What fraud-related questions can I answer for you?",
            "good evening": "Good evening! I'm Fina, ready to help with fraud detection insights.",
            "how are you": "I'm doing great, thank you! I'm Fina, your fraud detection assistant. How can I help you analyze fraud patterns today?",
            "thanks": "You're welcome! Feel free to ask me anything else about fraud detection.",
            "thank you": "You're very welcome! Let me know if you need help with anything else.",
            "bye": "Goodbye! Feel free to come back anytime you need fraud detection insights."
        }

        # Find matching greeting
        answer = None
        for key, response in greetings.items():
            if key in query:
                answer = response
                break

        # Default greeting if no match
        if not answer:
            answer = "Hello! I'm Fina, your fraud detection assistant. I can help you analyze transaction data and answer questions about fraud detection. What would you like to know?"

        state["answer"] = answer
        logger.info("Greeting handled")

        return state

    async def _extract_data_question(self, query: str) -> str:
        """Extract just the data/SQL-relevant part of a hybrid question."""
        prompt = f"""Extract ONLY the data-related question from the following query.
Remove any parts asking for explanations, methods, recommendations, or theoretical information.
Return only the factual data question that can be answered with SQL.

Original query: {query}

Examples:
- "What is our fraud rate by category and what fraud prevention methods should we use?"
  -> "What is the fraud rate by category?"
- "Show me top merchants with high fraud and explain why they are vulnerable"
  -> "Show me top merchants with high fraud"
- "What are our monthly fraud trends and how can we prevent fraud?"
  -> "What are our monthly fraud trends?"

Return ONLY the extracted data question, nothing else."""

        try:
            extracted = await llm_service.generate(prompt, temperature=0.0, max_tokens=100)
            extracted = extracted.strip().strip('"').strip("'")
            logger.info(f"Extracted data question: '{extracted}' from '{query}'")
            return extracted
        except Exception as e:
            logger.error(f"Error extracting data question: {e}")
            return query

    async def _run_sql_agent(self, state: AgentState) -> AgentState:
        """SQL agent node: Execute SQL query via Vanna."""
        query = state["user_query"]

        # For hybrid queries, extract just the data question
        if state.get("query_type") == "hybrid":
            query = await self._extract_data_question(query)
            state["extracted_sql_question"] = query

        logger.info(f"Running SQL agent for: {query}")

        try:
            # Use Vanna to generate and execute SQL
            result = await vanna_service.ask_async(query)

            if result["error"]:
                logger.error(f"SQL error: {result['error']}")
                state["sql_result"] = None
                state["error"] = result["error"]
            else:
                state["sql_result"] = {
                    "sql": result["sql"],
                    "data": result["results"]
                }
                logger.info(f"SQL executed successfully: {result['sql']}")
                logger.info(f"Returned {len(result['results'])} rows")

        except Exception as e:
            logger.error(f"Error in SQL agent: {e}")
            state["sql_result"] = None
            state["error"] = str(e)

        return state

    async def _extract_search_keywords(self, query: str) -> list:
        """Extract key search concepts/keywords from user question using LLM."""
        prompt = f"""Extract 3-5 key search keywords/phrases from the following question for document retrieval.
Focus on the main concepts that would help find relevant information in research documents.
Return as a JSON array of strings.

Question: {query}

Examples:
- "What are the primary methods by which credit card fraud is committed?"
  -> ["credit card fraud methods", "fraud techniques", "fraud types"]
- "What fraud prevention methods should we use?"
  -> ["fraud prevention methods", "fraud detection techniques", "prevention strategies"]
- "How does card-not-present fraud work?"
  -> ["card-not-present fraud", "CNP fraud", "online fraud mechanism"]

Return ONLY the JSON array, nothing else."""

        try:
            result = await llm_service.extract_json(prompt)
            if isinstance(result, list):
                logger.info(f"Extracted search keywords: {result}")
                return result
            return [query]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return [query]

    async def _evaluate_chunk_relevance(self, query: str, chunks: list) -> dict:
        """Evaluate if retrieved chunks are relevant and sufficient to answer the question."""
        if not chunks:
            return {"is_sufficient": False, "relevance_score": 0, "missing_info": query}

        chunks_preview = "\n\n".join([f"[Chunk {i+1}]: {c['text'][:500]}..." for i, c in enumerate(chunks[:5])])

        prompt = f"""Evaluate if these document chunks can answer the user's question.

User Question: {query}

Retrieved Chunks:
{chunks_preview}

Evaluate and respond with JSON:
{{
    "is_sufficient": true/false,
    "relevance_score": 0.0-1.0,
    "missing_info": "what information is missing if not sufficient",
    "refined_query": "a better search query if needed"
}}

Rules:
- is_sufficient = true if chunks contain enough info to answer the question meaningfully
- relevance_score = how relevant the chunks are (0.8+ is good)
- If not sufficient, suggest what's missing and a refined query"""

        try:
            result = await llm_service.extract_json(prompt)
            logger.info(f"Chunk relevance evaluation: sufficient={result.get('is_sufficient')}, score={result.get('relevance_score')}")
            return result
        except Exception as e:
            logger.error(f"Error evaluating chunks: {e}")
            return {"is_sufficient": True, "relevance_score": 0.5, "missing_info": ""}

    async def _run_rag_agent(self, state: AgentState) -> AgentState:
        """RAG agent node: Retrieve relevant documents with iterative refinement."""
        original_query = state["user_query"]
        query = original_query

        # For hybrid queries, extract the knowledge/theory part first
        if state.get("query_type") == "hybrid":
            prompt = f"""Extract ONLY the knowledge/theory question from this hybrid query.
Remove data-related parts (numbers, statistics, "our data", etc).

Query: {original_query}

Example:
- "What is our fraud rate by category and what fraud prevention methods should we use?"
  -> "What fraud prevention methods should we use?"

Return ONLY the knowledge question:"""
            try:
                query = await llm_service.generate(prompt, temperature=0.0, max_tokens=100)
                query = query.strip().strip('"').strip("'")
                state["extracted_rag_question"] = query
                logger.info(f"Extracted knowledge question: '{query}'")
            except:
                pass

        logger.info(f"Running RAG agent for: {query}")

        max_iterations = 2
        all_results = []
        tried_queries = set()

        try:
            for iteration in range(max_iterations):
                logger.info(f"RAG iteration {iteration + 1}/{max_iterations}")

                # Extract search keywords for this iteration
                if iteration == 0:
                    search_keywords = await self._extract_search_keywords(query)
                else:
                    # Use refined query from evaluation
                    search_keywords = [query]

                # Search with each keyword and combine results
                iteration_results = []
                for keyword in search_keywords:
                    if keyword in tried_queries:
                        continue
                    tried_queries.add(keyword)

                    query_embedding = await embedding_service.embed(keyword)
                    results = await elasticsearch_tool.hybrid_search(
                        query=keyword,
                        query_embedding=query_embedding,
                        top_k=3
                    )
                    iteration_results.extend(results)

                # Deduplicate by text content
                seen_texts = set()
                for r in iteration_results:
                    text_key = r["text"][:200]
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_results.append(r)

                # Sort by score and keep top results
                all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)[:7]

                if not all_results:
                    logger.warning(f"No results found in iteration {iteration + 1}")
                    continue

                # Evaluate if we have sufficient information
                evaluation = await self._evaluate_chunk_relevance(original_query, all_results)

                if evaluation.get("is_sufficient", False) or evaluation.get("relevance_score", 0) >= 0.7:
                    logger.info(f"Found sufficient chunks (score: {evaluation.get('relevance_score')})")
                    break

                # Not sufficient, try refined query in next iteration
                refined_query = evaluation.get("refined_query", "")
                if refined_query and refined_query != query:
                    query = refined_query
                    logger.info(f"Refining search with: '{refined_query}'")
                else:
                    break

            if all_results:
                # Keep top 5 most relevant chunks
                final_results = all_results[:5]
                context = "\n\n".join([r["text"] for r in final_results])
                state["rag_context"] = context
                state["rag_sources"] = final_results
                logger.info(f"RAG complete: {len(final_results)} chunks retrieved")
            else:
                logger.warning("No documents retrieved after all iterations")
                state["rag_context"] = None
                state["rag_sources"] = []

        except Exception as e:
            logger.error(f"Error in RAG agent: {e}")
            state["rag_context"] = None
            state["rag_sources"] = []
            state["error"] = str(e)

        return state

    async def _run_hybrid_agent(self, state: AgentState) -> AgentState:
        """Hybrid agent node: Fetch from BOTH SQL and RAG in parallel."""
        logger.info("Running hybrid agent (SQL + RAG in parallel)")

        try:
            # Execute both agents in parallel
            # Pass query_type to both agents so they know to extract relevant questions
            sql_state = state.copy()
            sql_state["query_type"] = "hybrid"
            rag_state = state.copy()
            rag_state["query_type"] = "hybrid"
            sql_task = self._run_sql_agent(sql_state)
            rag_task = self._run_rag_agent(rag_state)

            results = await asyncio.gather(sql_task, rag_task, return_exceptions=True)

            # Merge results
            sql_state = results[0] if not isinstance(results[0], Exception) else {}
            rag_state = results[1] if not isinstance(results[1], Exception) else {}

            # Combine both results into state
            state["sql_result"] = sql_state.get("sql_result")
            state["rag_context"] = rag_state.get("rag_context")
            state["rag_sources"] = rag_state.get("rag_sources", [])

            logger.info("Hybrid agent completed")

        except Exception as e:
            logger.error(f"Error in hybrid agent: {e}")
            state["error"] = str(e)

        return state

    async def _generate_answer(self, state: AgentState) -> AgentState:
        """Answer generation node: Generate final answer using LLM."""
        query = state["user_query"]
        query_type = state["query_type"]
        sql_result = state.get("sql_result")
        rag_context = state.get("rag_context")
        rag_sources = state.get("rag_sources", [])
        conversation_history = state.get("conversation_history", [])

        logger.info(f"Generating answer for query type: {query_type}")

        try:
            if query_type == "sql":
                # SQL-based answer
                prompt = self._build_sql_prompt(query, sql_result, conversation_history)
            elif query_type == "rag":
                # RAG-based answer with citations
                prompt = self._build_rag_prompt(query, rag_context, rag_sources, conversation_history)
            else:  # hybrid
                # Hybrid answer with citations
                prompt = self._build_hybrid_prompt(query, sql_result, rag_context, rag_sources, conversation_history)

            # Generate answer
            answer = await llm_service.generate(
                prompt=prompt,
                system_prompt="You are Fina, a helpful fraud detection assistant. Provide well-structured, readable answers using Markdown formatting (headers, bullets, bold, blockquotes). Keep paragraphs short. Use proper citations [1], [2], etc. DO NOT use emoticons or emojis.",
                temperature=0.1,
                max_tokens=2000  # Increased for better formatted answers
            )

            state["answer"] = answer
            logger.info(f"Generated answer ({len(answer)} characters)")

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["answer"] = f"I encountered an error generating the answer: {str(e)}"

        return state

    def _format_conversation_history(self, history: list) -> str:
        """Format conversation history for prompt inclusion."""
        if not history:
            return ""

        formatted = "\n\nPrevious Conversation:\n"
        for msg in history[-6:]:  # Last 6 messages (3 turns) for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted += f"{role.upper()}: {content}\n"

        formatted += "\n---\n"
        return formatted

    def _build_sql_prompt(self, query: str, sql_result: Dict, history: list = None) -> str:
        """Build prompt for SQL-based answer."""
        if not sql_result or not sql_result.get("data"):
            return f"""{self._format_conversation_history(history)}User Question: {query}

Unfortunately, I couldn't retrieve data from the database. Please provide a helpful response explaining this."""

        # Limit data to first 20 rows for token efficiency
        data = sql_result["data"][:20]

        return f"""{self._format_conversation_history(history)}Answer this question using our TRANSACTION DATA:

User Question: {query}

SQL Query Executed:
{sql_result.get("sql", "N/A")}

Transaction Data (showing up to 20 rows):
{json.dumps(data, indent=2, default=str)}

Instructions:
- Present factual findings from our database
- Use phrases like "In our data...", "Our transactions show...", "Based on our records..."
- Include specific numbers, dates, merchants from the data
- Be precise and factual
- If data is limited, acknowledge it

FORMATTING (IMPORTANT for readability):
- Use **bold** for key metrics and numbers
- Use bullet points for lists
- Use ## headers to organize content if answer has multiple parts (choose natural headers based on the data)
- Keep paragraphs short (2-3 sentences max)
- Add blank lines between sections for better readability

Create headers that naturally fit the data insights. DO NOT use emoticons or emojis.
"""

    def _build_rag_prompt(self, query: str, rag_context: str, rag_sources: list = None, history: list = None) -> str:
        """Build prompt for RAG-based answer with numbered citations."""
        if not rag_context:
            return f"""{self._format_conversation_history(history)}User Question: {query}

Unfortunately, I couldn't find relevant information in the fraud detection literature. Please provide a general helpful response."""

        # Build numbered sources for citation
        sources_text = ""
        if rag_sources:
            for i, source in enumerate(rag_sources, 1):
                metadata = source.get("metadata", {})
                filename = metadata.get("filename", "Unknown")
                page = metadata.get("page", "N/A")
                sources_text += f"\n[{i}] {filename} (Page {page})\n{source['text']}\n"
        else:
            sources_text = rag_context

        return f"""{self._format_conversation_history(history)}Answer this question using FRAUD DETECTION RESEARCH & THEORY:

User Question: {query}

Relevant Document Sources:
{sources_text}

Instructions:
- Explain concepts, methods, theories from the research
- **IMPORTANT**: Use inline citations [1], [2], [3] etc. when referencing information from sources
- Example: "According to fraud detection research, card-not-present transactions have higher fraud rates [1]."
- Use phrases like "According to fraud detection research...", "The document states...", "Industry standards suggest..."
- Focus on WHY and HOW, providing conceptual understanding
- Always cite which source number supports each claim

FORMATTING (IMPORTANT for readability):
- Use ## headers to organize content (choose headers that naturally fit the topic)
- Use bullet points for listing methods, techniques, components
- Use **bold** for key concepts and important terms
- Use > blockquotes for important research findings or quotes
- Keep paragraphs short (2-4 sentences max)
- Add blank lines between sections for better readability
- Use ### sub-headers when breaking down complex concepts

Create natural, contextual headers based on the question content.
DO NOT use emoticons or emojis.
"""

    def _build_hybrid_prompt(self, query: str, sql_result: Dict, rag_context: str, rag_sources: list = None, history: list = None) -> str:
        """Build prompt for hybrid answer with numbered citations."""
        # Build SQL data section
        sql_section = "No transaction data available"
        if sql_result and sql_result.get("data"):
            data = sql_result["data"][:15]
            sql_section = f"""SQL Query: {sql_result.get("sql", "N/A")}

Transaction Data:
{json.dumps(data, indent=2, default=str)}"""

        # Build RAG section with numbered sources
        rag_section = "No research context available"
        if rag_context and rag_sources:
            rag_section = ""
            for i, source in enumerate(rag_sources, 1):
                metadata = source.get("metadata", {})
                filename = metadata.get("filename", "Unknown")
                page = metadata.get("page", "N/A")
                rag_section += f"\n[{i}] {filename} (Page {page})\n{source['text']}\n"
        elif rag_context:
            rag_section = rag_context

        return f"""{self._format_conversation_history(history)}Answer this question using BOTH our transaction data AND fraud research:

User Question: {query}

OUR TRANSACTION DATA (factual):
{sql_section}

FRAUD RESEARCH & THEORY (conceptual):
{rag_section}

Instructions:
1. FIRST: Present factual findings from our transaction data
   - "Our transaction data shows..."
   - Include specific numbers/merchants from database

2. THEN: Add theoretical context from research with citations
   - "According to fraud detection literature..."
   - **IMPORTANT**: Use inline citations [1], [2], [3] when referencing research
   - Example: "Card-not-present fraud is more common in e-commerce [1]."
   - Explain WHY patterns occur based on research

3. COMBINE: Link our data to theoretical frameworks
   - "This aligns with research that suggests... [2]"
   - Provide comprehensive analysis with proper citations

Make it clear what comes from OUR data vs EXTERNAL research.
Always cite source numbers when using research information.

FORMATTING (IMPORTANT for readability):
Structure your answer like a well-formatted blog post with natural, contextual headers.

FORMATTING GUIDELINES:
- Use ## headers to organize content into logical sections (choose headers that fit the question naturally)
- Use bullet points for lists and key findings
- **Bold** important numbers, metrics, and key terms
- Use > blockquotes for important research quotes or findings
- Keep paragraphs SHORT (2-4 sentences maximum)
- Add blank lines between sections for breathing room
- Use ### sub-headers when breaking down complex topics

EXAMPLE STRUCTURE (adapt headers to your specific content):
For a question about "fraud rate fluctuations":
## Monthly Fraud Rate Trends
## Seasonal Patterns Observed
## Contributing Factors

For a question about "fraud methods":
## Common Fraud Techniques
## How Each Method Works
## Prevention Strategies

For hybrid questions, combine data findings with research insights naturally.

DO NOT use emoticons or emojis.
DO NOT use generic template headers - make them specific to the question.
"""

    async def _calculate_quality_score(self, state: AgentState) -> AgentState:
        """Quality scoring node: Calculate answer quality metrics."""
        logger.info("Calculating quality score")

        try:
            scores = {}

            # 1. Relevance (semantic similarity)
            scores["relevance"] = await self._calculate_relevance(state)

            # 2. Data support
            scores["data_support"] = self._calculate_data_support(state)

            # 3. Confidence (from classification)
            scores["confidence"] = state.get("classification_confidence", 0.5)

            # 4. Completeness
            scores["completeness"] = self._calculate_completeness(state)

            # 5. Overall score (weighted average)
            scores["overall"] = (
                scores["relevance"] * 0.3 +
                scores["data_support"] * 0.3 +
                scores["confidence"] * 0.2 +
                scores["completeness"] * 0.2
            )

            state["quality_score"] = scores
            logger.info(f"Quality scores: {scores}")

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            state["quality_score"] = {
                "relevance": 0.5,
                "data_support": 0.5,
                "confidence": 0.5,
                "completeness": 0.5,
                "overall": 0.5
            }

        return state

    async def _calculate_relevance(self, state: AgentState) -> float:
        """Calculate semantic similarity between query and answer."""
        try:
            query = state["user_query"]
            answer = state.get("answer", "")

            if not answer:
                return 0.3

            # Generate embeddings
            query_emb = await embedding_service.embed(query)
            answer_emb = await embedding_service.embed(answer[:1000])  # Limit for efficiency

            # Cosine similarity
            similarity = np.dot(query_emb, answer_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(answer_emb)
            )

            # Normalize to 0-1
            relevance = (similarity + 1) / 2
            return float(relevance)

        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5

    def _calculate_data_support(self, state: AgentState) -> float:
        """Calculate whether answer has backing data."""
        query_type = state["query_type"]
        has_sql = state.get("sql_result") is not None and state.get("sql_result", {}).get("data")
        has_rag = state.get("rag_context") is not None

        if query_type == "greeting":
            # Greeting doesn't need data support
            return 1.0
        elif query_type == "hybrid":
            # Hybrid needs both
            if has_sql and has_rag:
                return 1.0
            elif has_sql or has_rag:
                return 0.6
            else:
                return 0.2
        elif query_type == "sql":
            return 1.0 if has_sql else 0.3
        else:  # rag
            return 1.0 if has_rag else 0.3

    def _calculate_completeness(self, state: AgentState) -> float:
        """Calculate answer completeness."""
        answer = state.get("answer", "")

        if not answer:
            return 0.0

        # Check length (200+ chars is complete)
        length_score = min(len(answer) / 200, 1.0)

        # Check for citations/sources
        citation_keywords = ["based on", "data shows", "according to", "research indicates", "our data"]
        has_citations = any(keyword in answer.lower() for keyword in citation_keywords)
        citation_score = 1.0 if has_citations else 0.5

        # Check for error messages
        has_error = "error" in answer.lower() or "couldn't" in answer.lower()
        error_penalty = 0.5 if has_error else 1.0

        completeness = (length_score + citation_score) / 2 * error_penalty
        return min(completeness, 1.0)

    async def process(self, user_query: str, conversation_id: str = None, conversation_history: list = None) -> Dict:
        """
        Process a user query through the agent workflow.

        Args:
            user_query: The user's question
            conversation_id: Optional conversation ID for tracking
            conversation_history: List of previous messages [{"role": "user|assistant", "content": "..."}]

        Returns:
            Dict containing answer, quality_score, and metadata
        """
        logger.info("=" * 60)
        logger.info(f"Processing query: {user_query}")
        logger.info("=" * 60)

        # Initialize state
        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history or [],
        }

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Build response
        response = {
            "answer": final_state.get("answer", "I couldn't generate an answer."),
            "quality_score": final_state.get("quality_score", {}),
            "query_type": final_state.get("query_type"),
            "classification_confidence": final_state.get("classification_confidence"),
            "sql_query": final_state.get("sql_result", {}).get("sql") if final_state.get("sql_result") else None,
            "sources": final_state.get("rag_sources", []),  # Return full source objects with metadata
            "error": final_state.get("error")
        }

        # Detailed evaluation logging
        self._log_evaluation(final_state, response)

        logger.info(f"Query processing complete. Quality score: {response['quality_score'].get('overall', 0):.2f}")

        return response

    def _log_evaluation(self, state: AgentState, response: Dict):
        """Log detailed evaluation information for analysis."""
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        # 1. Query Classification
        logger.info(f"\n[CLASSIFICATION]")
        logger.info(f"  Type: {state.get('query_type')}")
        logger.info(f"  Confidence: {state.get('classification_confidence', 0):.2f}")
        logger.info(f"  Reasoning: {state.get('classification_reasoning', 'N/A')}")

        # 2. SQL Agent Results (if applicable)
        if state.get('sql_result'):
            sql_result = state['sql_result']
            logger.info(f"\n[SQL AGENT]")
            logger.info(f"  Query: {sql_result.get('sql', 'N/A')}")
            data = sql_result.get('data', [])
            logger.info(f"  Rows Returned: {len(data)}")
            if data:
                logger.info(f"  Sample Data (first 3 rows):")
                for i, row in enumerate(data[:3], 1):
                    logger.info(f"    Row {i}: {row}")

        # 3. RAG Agent Results (if applicable)
        if state.get('rag_sources'):
            logger.info(f"\n[RAG AGENT]")
            logger.info(f"  Chunks Retrieved: {len(state['rag_sources'])}")
            for i, source in enumerate(state['rag_sources'], 1):
                logger.info(f"  Chunk {i}:")
                logger.info(f"    Score: {source.get('score', 0):.4f}")
                logger.info(f"    Source: {source.get('metadata', {}).get('filename', 'N/A')}")
                logger.info(f"    Preview: {source.get('text', '')[:150]}...")

        # 4. Answer Generation
        answer = state.get('answer', '')
        logger.info(f"\n[ANSWER]")
        logger.info(f"  Length: {len(answer)} characters")
        logger.info(f"  Preview: {answer[:200]}...")

        # 5. Quality Scores
        quality = response.get('quality_score', {})
        logger.info(f"\n[QUALITY SCORES]")
        logger.info(f"  Relevance: {quality.get('relevance', 0):.2f}")
        logger.info(f"  Data Support: {quality.get('data_support', 0):.2f}")
        logger.info(f"  Confidence: {quality.get('confidence', 0):.2f}")
        logger.info(f"  Completeness: {quality.get('completeness', 0):.2f}")
        logger.info(f"  Overall: {quality.get('overall', 0):.2f}")

        # 6. Errors (if any)
        if state.get('error'):
            logger.warning(f"\n[ERRORS]")
            logger.warning(f"  Error: {state['error']}")

        logger.info("=" * 60 + "\n")

    async def process_stream(self, user_query: str, conversation_id: str = None, conversation_history: list = None):
        """
        Process query with streaming updates.

        Args:
            user_query: The user's question
            conversation_id: Optional conversation ID
            conversation_history: List of previous messages for context

        Yields progress updates and final answer chunks.
        """
        logger.info("=" * 60)
        logger.info(f"Processing query (streaming): {user_query}")
        logger.info("=" * 60)

        # Yield start event
        yield {
            "type": "start",
            "query": user_query
        }

        # Initialize state
        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history or [],
        }

        try:
            # Step 1: Classification
            yield {
                "type": "progress",
                "step": "classification",
                "message": "Classifying query intent..."
            }

            state = await self._route_query(initial_state)

            yield {
                "type": "classification",
                "query_type": state.get("query_type"),
                "confidence": state.get("classification_confidence"),
                "reasoning": state.get("classification_reasoning")
            }

            # Step 2: Data Collection
            query_type = state["query_type"]

            if query_type == "greeting":
                yield {
                    "type": "progress",
                    "step": "greeting",
                    "message": "Generating greeting..."
                }
                state = await self._handle_greeting(state)

            elif query_type == "sql":
                yield {
                    "type": "progress",
                    "step": "sql",
                    "message": "Generating and executing SQL query..."
                }
                state = await self._run_sql_agent(state)

                if state.get("sql_result"):
                    yield {
                        "type": "sql_result",
                        "sql": state["sql_result"].get("sql"),
                        "rows": len(state["sql_result"].get("data", []))
                    }

            elif query_type == "rag":
                yield {
                    "type": "progress",
                    "step": "rag",
                    "message": "Searching fraud detection documents..."
                }
                state = await self._run_rag_agent(state)

                if state.get("rag_sources"):
                    yield {
                        "type": "rag_result",
                        "chunks": len(state["rag_sources"]),
                        "sources": state["rag_sources"]  # Return full source objects
                    }

            elif query_type == "hybrid":
                yield {
                    "type": "progress",
                    "step": "hybrid",
                    "message": "Fetching data from database and documents..."
                }
                state = await self._run_hybrid_agent(state)

                results = {}
                if state.get("sql_result"):
                    results["sql_rows"] = len(state["sql_result"].get("data", []))
                if state.get("rag_sources"):
                    results["rag_chunks"] = len(state["rag_sources"])

                yield {
                    "type": "hybrid_result",
                    **results
                }

            # Step 3: Answer Generation
            if query_type != "greeting":
                yield {
                    "type": "progress",
                    "step": "answer_generation",
                    "message": "Generating answer..."
                }

                # Generate answer with streaming
                state = await self._generate_answer(state)

            # Stream answer chunks
            answer = state.get("answer", "")
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                yield {
                    "type": "answer_chunk",
                    "content": answer[i:i+chunk_size]
                }

            # Step 4: Quality Scoring
            yield {
                "type": "progress",
                "step": "quality_scoring",
                "message": "Calculating quality score..."
            }

            state = await self._calculate_quality_score(state)

            yield {
                "type": "quality_score",
                "scores": state.get("quality_score", {})
            }

            # Final event
            yield {
                "type": "done",
                "query_type": state.get("query_type"),
                "sql_query": state.get("sql_result", {}).get("sql") if state.get("sql_result") else None,
                "sources": state.get("rag_sources", [])  # Return full source objects
            }

            # Log evaluation (keep filenames for logging only)
            response = {
                "answer": answer,
                "quality_score": state.get("quality_score", {}),
                "query_type": state.get("query_type"),
                "classification_confidence": state.get("classification_confidence"),
                "sql_query": state.get("sql_result", {}).get("sql") if state.get("sql_result") else None,
                "sources": [s.get("metadata", {}).get("filename") for s in state.get("rag_sources", [])],  # Just filenames for logging
                "error": state.get("error")
            }
            self._log_evaluation(state, response)

        except Exception as e:
            logger.error(f"Error in streaming process: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }


# Singleton instance
fina_orchestrator = FinaOrchestrator()
