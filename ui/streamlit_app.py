"""
Streamlit UI for Fina - Fraud Detection Chatbot Assistant
Professional design without emoticons
"""
import streamlit as st
import requests
import json
import os
from typing import Dict, List

# Page config
st.set_page_config(
    page_title="Fina | Fraud Detection Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        color: white;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }

    .app-subtitle {
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        color: #e0e7ff !important;
    }

    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-sql {
        background-color: #dbeafe;
        color: #1e40af !important;
    }

    .status-rag {
        background-color: #fef3c7;
        color: #92400e !important;
    }

    .status-hybrid {
        background-color: #e0e7ff;
        color: #4338ca !important;
    }

    .status-greeting {
        background-color: #d1fae5;
        color: #065f46 !important;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint - use environment variable or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000/api")


def get_conversations() -> List[Dict]:
    """Fetch conversation list from API."""
    try:
        response = requests.get(f"{API_URL}/conversations")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return []


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation."""
    try:
        response = requests.delete(f"{API_URL}/conversations/{conversation_id}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        return False


def get_sample_questions() -> List[Dict]:
    """Fetch sample questions from API."""
    try:
        response = requests.get(f"{API_URL}/sample-questions")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching sample questions: {e}")
        return []


def send_chat_message_stream(query: str, conversation_id: str = None):
    """Send chat message to API with streaming."""
    try:
        payload = {"query": query, "stream": True}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )

        if response.status_code == 200:
            # Parse SSE events
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        try:
                            event = json.loads(data)
                            yield event
                        except json.JSONDecodeError:
                            continue
        else:
            yield {"type": "error", "error": f"API error: {response.status_code}"}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


def display_quality_score(score: Dict):
    """Display quality score metrics in a professional way."""
    overall = score.get("overall", 0)

    # Color based on score
    if overall >= 0.8:
        color = "#10b981"
        label = "Excellent"
    elif overall >= 0.6:
        color = "#f59e0b"
        label = "Good"
    else:
        color = "#ef4444"
        label = "Fair"

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; font-size: 0.875rem; color: #64748b;">QUALITY SCORE</span>
            <span style="background-color: {color}; color: white; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 600;">{label}</span>
        </div>
        <div style="background-color: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {overall*100}%; transition: width 0.3s ease;"></div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-top: 0.5rem;">{overall:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Show breakdown in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Relevance", f"{score.get('relevance', 0):.2f}", delta=None)
    with col2:
        st.metric("Data Support", f"{score.get('data_support', 0):.2f}", delta=None)
    with col3:
        st.metric("Confidence", f"{score.get('confidence', 0):.2f}", delta=None)
    with col4:
        st.metric("Completeness", f"{score.get('completeness', 0):.2f}", delta=None)


def get_query_type_badge(query_type: str) -> str:
    """Get HTML badge for query type."""
    badges = {
        "greeting": '<span class="status-badge status-greeting">Greeting</span>',
        "sql": '<span class="status-badge status-sql">Data Analysis</span>',
        "rag": '<span class="status-badge status-rag">Knowledge Base</span>',
        "hybrid": '<span class="status-badge status-hybrid">Hybrid Analysis</span>'
    }
    return badges.get(query_type, '<span class="status-badge">Unknown</span>')


# Header
conversation_subtitle = st.session_state.get("conversation_title", "New Conversation")
st.markdown(f"""
<div class="app-header">
    <h1 class="app-title">Fina</h1>
    <p class="app-subtitle">AI-Powered Fraud Detection Assistant | {conversation_subtitle}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Conversation Management
    st.markdown("### Conversations")

    # New Chat button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
        st.session_state.conversation_title = "New Conversation"
        st.rerun()

    st.markdown("---")

    # List conversations
    conversations = get_conversations()
    if conversations:
        st.markdown("**Recent Conversations**")
        for conv in conversations[:10]:  # Show 10 most recent
            conv_title = conv.get("title", "Untitled")[:40]
            if len(conv.get("title", "")) > 40:
                conv_title += "..."

            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(conv_title, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state.current_conversation_id = conv["id"]
                    st.session_state.conversation_title = conv.get("title", "Untitled")
                    # Clear current messages, they'll be loaded from conversation
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("X", key=f"del_{conv['id']}"):
                    if delete_conversation(conv["id"]):
                        if st.session_state.current_conversation_id == conv["id"]:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.rerun()

    st.markdown("---")

    # Sample Questions (collapsed by default)
    with st.expander("Sample Questions"):
        samples = get_sample_questions()

        # Group by type
        sql_questions = [s for s in samples if s.get("type") == "sql"]
        rag_questions = [s for s in samples if s.get("type") == "rag"]
        hybrid_questions = [s for s in samples if s.get("type") == "hybrid"]

        if sql_questions:
            st.markdown("**Data Analysis**")
            for i, sample in enumerate(sql_questions[:2]):
                if st.button(sample["question"][:50] + "..." if len(sample["question"]) > 50 else sample["question"],
                            key=f"sql_{i}", use_container_width=True):
                    st.session_state.current_query = sample["question"]
                    st.rerun()

        if rag_questions:
            st.markdown("**Knowledge Base**")
            for i, sample in enumerate(rag_questions[:2]):
                if st.button(sample["question"][:50] + "..." if len(sample["question"]) > 50 else sample["question"],
                            key=f"rag_{i}", use_container_width=True):
                    st.session_state.current_query = sample["question"]
                    st.rerun()

        if hybrid_questions:
            st.markdown("**Hybrid Analysis**")
            for i, sample in enumerate(hybrid_questions[:2]):
                if st.button(sample["question"][:50] + "..." if len(sample["question"]) > 50 else sample["question"],
                            key=f"hybrid_{i}", use_container_width=True):
                    st.session_state.current_query = sample["question"]
                    st.rerun()

    # About section
    with st.expander("About"):
        st.markdown("""
        **Fina** combines transaction data analysis with fraud detection research to provide comprehensive insights.

        **Features:**
        - SQL-based transaction analysis
        - RAG over fraud detection research
        - LangGraph orchestration
        - Real-time streaming responses
        - Conversation history
        """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "process_query" not in st.session_state:
    st.session_state.process_query = None

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = "New Conversation"

# Handle sample question click
if "current_query" in st.session_state and st.session_state.current_query:
    query = st.session_state.current_query
    st.session_state.current_query = None
    st.session_state.process_query = query

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]

            st.markdown("---")

            # Quality score
            if "quality_score" in metadata and metadata["quality_score"]:
                display_quality_score(metadata["quality_score"])
                st.markdown("")

            # Query type and classification
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Query Type**")
                st.markdown(get_query_type_badge(metadata.get("query_type", "unknown")), unsafe_allow_html=True)
            with col2:
                st.markdown("**Classification Confidence**")
                confidence = metadata.get("classification_confidence", 0)
                st.progress(confidence, text=f"{confidence:.0%}")

            # Classification reasoning
            if metadata.get("classification_reasoning"):
                with st.expander("Classification Reasoning"):
                    st.info(metadata["classification_reasoning"])

            # SQL query if available
            if metadata.get("sql_query"):
                with st.expander("SQL Query"):
                    st.code(metadata["sql_query"], language="sql")

            # Sources if available (with citations)
            if metadata.get("sources"):
                with st.expander(f"References ({len(metadata['sources'])} sources)"):
                    for i, source in enumerate(metadata["sources"], 1):
                        # Extract metadata
                        source_metadata = source.get("metadata", {})
                        filename = source_metadata.get("filename", "Unknown")
                        page = source_metadata.get("page", "N/A")
                        chunk_text = source.get("text", "")
                        score = source.get("score", 0)

                        # Create document URL
                        doc_url = f"http://localhost:8000/documents/{filename}"

                        st.markdown(f"**[{i}]** [{filename}]({doc_url}) (Page {page})")
                        st.caption(f"Relevance: {score:.3f}")

                        # Show chunk preview (first 500 chars for better context)
                        preview = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                        st.markdown(f"> {preview}")
                        st.markdown("---")

            # Error if any
            if metadata.get("error"):
                st.error(metadata["error"])

# Check if we need to process a query (from sample question or chat input)
prompt = st.session_state.process_query or st.chat_input("Ask about fraud patterns, transaction data, or detection methods...")

if prompt:
    # Clear process_query flag
    if st.session_state.process_query:
        st.session_state.process_query = None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API with streaming
    with st.chat_message("assistant"):
        # Placeholders for dynamic updates
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        metadata_container = st.container()

        # Variables to collect data
        answer_text = ""
        metadata = {
            "query_type": "unknown",
            "classification_confidence": 0,
            "classification_reasoning": "",
            "sql_query": None,
            "sources": [],
            "quality_score": None
        }

        try:
            # Stream events
            for event in send_chat_message_stream(prompt, st.session_state.current_conversation_id):
                event_type = event.get("type")

                if event_type == "start":
                    status_placeholder.info("Initializing analysis...")

                elif event_type == "progress":
                    step = event.get("step", "")
                    message = event.get("message", "")
                    step_name = step.replace('_', ' ').title()
                    status_placeholder.info(f"**{step_name}** • {message}")

                elif event_type == "classification":
                    metadata["query_type"] = event.get("query_type", "unknown")
                    metadata["classification_confidence"] = event.get("confidence", 0)
                    metadata["classification_reasoning"] = event.get("reasoning", "")

                    query_type_text = metadata["query_type"].replace('_', ' ').title()
                    status_placeholder.success(f"Classified as **{query_type_text}** (confidence: {metadata['classification_confidence']:.0%})")

                elif event_type == "sql_result":
                    metadata["sql_query"] = event.get("sql")
                    rows = event.get("rows", 0)
                    status_placeholder.success(f"SQL query executed • {rows} rows returned")

                elif event_type == "rag_result":
                    chunks = event.get("chunks", 0)
                    metadata["sources"] = event.get("sources", [])
                    status_placeholder.success(f"Retrieved {chunks} relevant document chunks")

                elif event_type == "hybrid_result":
                    sql_rows = event.get("sql_rows", 0)
                    rag_chunks = event.get("rag_chunks", 0)
                    status_placeholder.success(f"Fetched {sql_rows} database rows + {rag_chunks} document chunks")

                elif event_type == "answer_chunk":
                    answer_text += event.get("content", "")
                    answer_placeholder.markdown(answer_text + "█")  # Block cursor

                elif event_type == "quality_score":
                    metadata["quality_score"] = event.get("scores", {})

                elif event_type == "done":
                    # Update final metadata
                    if event.get("sql_query"):
                        metadata["sql_query"] = event["sql_query"]
                    if event.get("sources"):
                        metadata["sources"] = event["sources"]

                    # Clear status and show final answer
                    status_placeholder.empty()
                    answer_placeholder.markdown(answer_text)

                elif event_type == "error":
                    status_placeholder.error(f"Error: {event.get('error')}")
                    answer_text = "I encountered an error processing your request. Please try again."
                    answer_placeholder.markdown(answer_text)

            # Display metadata after streaming completes
            with metadata_container:
                st.markdown("---")

                if metadata.get("quality_score"):
                    display_quality_score(metadata["quality_score"])
                    st.markdown("")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Query Type**")
                    st.markdown(get_query_type_badge(metadata["query_type"]), unsafe_allow_html=True)
                with col2:
                    st.markdown("**Classification Confidence**")
                    confidence = metadata["classification_confidence"]
                    st.progress(confidence, text=f"{confidence:.0%}")

                if metadata.get("classification_reasoning"):
                    with st.expander("Classification Reasoning"):
                        st.info(metadata["classification_reasoning"])

                if metadata.get("sql_query"):
                    with st.expander("SQL Query"):
                        st.code(metadata["sql_query"], language="sql")

                if metadata.get("sources"):
                    with st.expander(f"References ({len(metadata['sources'])} sources)"):
                        for i, source in enumerate(metadata["sources"], 1):
                            # Extract metadata
                            source_metadata = source.get("metadata", {})
                            filename = source_metadata.get("filename", "Unknown")
                            page = source_metadata.get("page", "N/A")
                            chunk_text = source.get("text", "")
                            score = source.get("score", 0)

                            # Create document URL
                            doc_url = f"http://localhost:8000/documents/{filename}"

                            st.markdown(f"**[{i}]** [{filename}]({doc_url}) (Page {page})")
                            st.caption(f"Relevance: {score:.3f}")

                            # Show chunk preview (first 500 chars for better context)
                            preview = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                            st.markdown(f"> {preview}")
                            st.markdown("---")

        except Exception as e:
            st.error(f"Streaming error: {e}")
            answer_text = "I encountered an error during streaming. Please try again."

        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": metadata
        })
