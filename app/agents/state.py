"""
Agent state definition for LangGraph workflow.
"""
from typing import TypedDict, Optional, List, Dict, Literal


class AgentState(TypedDict, total=False):
    """State for Fina chatbot agent workflow."""

    # Input
    user_query: str
    conversation_id: Optional[str]
    conversation_history: List[Dict]  # Previous messages for context

    # Query classification
    query_type: Literal["greeting", "sql", "rag", "hybrid"]
    classification_confidence: float
    classification_reasoning: str

    # Extracted questions for hybrid queries
    extracted_sql_question: Optional[str]  # Data part extracted for SQL
    extracted_rag_question: Optional[str]  # Knowledge part extracted for RAG

    # Data sources
    sql_result: Optional[Dict]  # {sql: str, data: List[Dict]}
    rag_context: Optional[str]
    rag_sources: List[Dict]  # [{text: str, metadata: Dict, score: float}]

    # Output
    answer: str
    quality_score: Dict[str, float]  # {relevance, data_support, confidence, completeness, overall}

    # Metadata
    error: Optional[str]
