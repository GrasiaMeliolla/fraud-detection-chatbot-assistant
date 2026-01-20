"""
Chat API endpoints for Fina chatbot.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload
from app.agents.orchestrator import fina_orchestrator
from app.db.session import get_async_session, AsyncSessionLocal
from app.db.models import Conversation, Message
import logging
import json
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User's question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    stream: bool = Field(False, description="Enable streaming response")


class QualityScore(BaseModel):
    """Quality score model."""
    relevance: float
    data_support: float
    confidence: float
    completeness: float
    overall: float


class Source(BaseModel):
    """Source document model."""
    text: str
    metadata: Dict
    score: float


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    quality_score: QualityScore
    query_type: str
    classification_confidence: float
    sql_query: Optional[str] = None
    sources: List[Source] = []
    error: Optional[str] = None


class SampleQuestion(BaseModel):
    """Sample question model."""
    question: str
    type: str


class ConversationListItem(BaseModel):
    """Conversation list item model."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


async def get_or_create_conversation(conversation_id: Optional[str], db: AsyncSession) -> Conversation:
    """Get existing conversation or create new one."""
    if conversation_id:
        # Load existing conversation
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    else:
        # Create new conversation
        conversation = Conversation(id=str(uuid.uuid4()))
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation


async def load_conversation_history(conversation_id: str, db: AsyncSession) -> List[Dict]:
    """Load conversation history from database."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    history = []
    for msg in messages:
        history.append({
            "role": msg.role,
            "content": msg.content
        })
    return history


async def save_messages(conversation: Conversation, user_query: str, assistant_response: Dict, db: AsyncSession):
    """Save user and assistant messages to database."""
    # Save user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        role="user",
        content=user_query
    )
    db.add(user_msg)

    # Save assistant message with metadata
    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_response["answer"],
        message_metadata={
            "query_type": assistant_response.get("query_type"),
            "classification_confidence": assistant_response.get("classification_confidence"),
            "sql_query": assistant_response.get("sql_query"),
            "quality_score": assistant_response.get("quality_score"),
            "sources": [{"text": s.get("text", ""), "metadata": s.get("metadata", {}), "score": s.get("score", 0)} for s in assistant_response.get("sources", [])]
        }
    )
    db.add(assistant_msg)

    # Update conversation title if first message
    if not conversation.title:
        # Use first 50 chars of user query as title
        conversation.title = user_query[:50] + ("..." if len(user_query) > 50 else "")

    await db.commit()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_async_session)):
    """
    Chat endpoint for asking questions to Fina.

    Args:
        request: Chat request with query and optional conversation_id

    Returns:
        ChatResponse with answer, quality score, and metadata
        Or StreamingResponse if stream=True
    """
    try:
        logger.info(f"Received chat request: {request.query} (stream={request.stream})")

        # Get or create conversation
        conversation = await get_or_create_conversation(request.conversation_id, db)

        # Load conversation history
        history = await load_conversation_history(conversation.id, db)

        # Streaming response
        if request.stream:
            async def generate():
                # Collect data for saving
                answer_text = ""
                metadata = {
                    "query_type": None,
                    "classification_confidence": 0,
                    "sql_query": None,
                    "quality_score": None,
                    "sources": []
                }

                async for event in fina_orchestrator.process_stream(
                    user_query=request.query,
                    conversation_id=conversation.id,
                    conversation_history=history
                ):
                    # Collect data from events
                    event_type = event.get("type")
                    if event_type == "classification":
                        metadata["query_type"] = event.get("query_type")
                        metadata["classification_confidence"] = event.get("confidence", 0)
                    elif event_type == "sql_result":
                        metadata["sql_query"] = event.get("sql")
                    elif event_type == "rag_result":
                        metadata["sources"] = event.get("sources", [])
                    elif event_type == "answer_chunk":
                        answer_text += event.get("content", "")
                    elif event_type == "quality_score":
                        metadata["quality_score"] = event.get("scores", {})
                    elif event_type == "done":
                        if event.get("sql_query"):
                            metadata["sql_query"] = event["sql_query"]
                        if event.get("sources"):
                            metadata["sources"] = event["sources"]

                    yield f"data: {json.dumps(event)}\n\n"

                # Save messages after streaming completes using a new session
                try:
                    async with AsyncSessionLocal() as save_db:
                        result = {
                            "answer": answer_text,
                            "query_type": metadata["query_type"],
                            "classification_confidence": metadata["classification_confidence"],
                            "sql_query": metadata["sql_query"],
                            "quality_score": metadata["quality_score"],
                            "sources": metadata["sources"]
                        }
                        await save_messages(conversation, request.query, result, save_db)
                        logger.info(f"Saved streaming messages for conversation {conversation.id}")
                except Exception as e:
                    logger.error(f"Error saving streaming messages: {e}")

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Conversation-Id": conversation.id,
                }
            )

        # Non-streaming response
        result = await fina_orchestrator.process(
            user_query=request.query,
            conversation_id=conversation.id,
            conversation_history=history
        )

        # Save messages to database
        await save_messages(conversation, request.query, result, db)

        # Build response
        response = ChatResponse(
            answer=result["answer"],
            quality_score=QualityScore(**result["quality_score"]),
            query_type=result["query_type"],
            classification_confidence=result["classification_confidence"],
            sql_query=result.get("sql_query"),
            sources=result.get("sources", []),
            error=result.get("error")
        )

        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-questions", response_model=List[SampleQuestion])
async def get_sample_questions():
    """
    Get sample questions that Fina can answer.

    Returns:
        List of sample questions with their types
    """
    samples = [
        {
            "question": "How does the daily or monthly fraud rate fluctuate over the two-year period?",
            "type": "sql"
        },
        {
            "question": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
            "type": "sql"
        },
        {
            "question": "What are the primary methods by which credit card fraud is committed?",
            "type": "rag"
        },
        {
            "question": "What are the core components of an effective fraud detection system, according to the authors?",
            "type": "rag"
        },
        {
            "question": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
            "type": "rag"
        },
        {
            "question": "What is the average transaction amount for fraudulent vs legitimate transactions?",
            "type": "sql"
        },
        {
            "question": "Which states have the highest fraud rates?",
            "type": "sql"
        },
        {
            "question": "Explain the different fraud techniques mentioned in fraud detection research",
            "type": "rag"
        },
        {
            "question": "Show me the top 10 merchants with highest fraud and explain why merchant fraud is common",
            "type": "hybrid"
        },
        {
            "question": "What is our fraud rate by category and what fraud prevention methods should we use?",
            "type": "hybrid"
        }
    ]

    return [SampleQuestion(**s) for s in samples]


@router.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations(db: AsyncSession = Depends(get_async_session)):
    """List all conversations ordered by most recent."""
    # Use subquery to count messages (avoids lazy loading issue with async SQLAlchemy)
    message_count_subq = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )

    result = await db.execute(
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            message_count_subq.label("message_count")
        )
        .order_by(desc(Conversation.updated_at))
        .limit(50)
    )
    rows = result.all()

    items = []
    for row in rows:
        items.append(ConversationListItem(
            id=row.id,
            title=row.title or "New Conversation",
            created_at=row.created_at,
            updated_at=row.updated_at,
            message_count=row.message_count or 0
        ))

    return items


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_async_session)):
    """Delete a conversation and all its messages."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()

    return {"status": "deleted", "conversation_id": conversation_id}


@router.get("/health")
async def health():
    """Health check for chat API."""
    return {"status": "healthy", "service": "chat"}
