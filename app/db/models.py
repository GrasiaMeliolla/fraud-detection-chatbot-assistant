from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, Date, Index, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class FraudTransaction(Base):
    """Fraud transaction model based on Kaggle fraud detection dataset."""

    __tablename__ = "fraud_transactions"

    # Primary key
    trans_num = Column(String(100), primary_key=True, index=True)

    # Transaction details
    trans_date_trans_time = Column(DateTime, nullable=False, index=True)
    unix_time = Column(BigInteger)
    amt = Column(Float, nullable=False)

    # Card details
    cc_num = Column(BigInteger, nullable=False, index=True)

    # Merchant details
    merchant = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    merch_lat = Column(Float)
    merch_long = Column(Float)

    # Cardholder details
    first = Column(String(100))
    last = Column(String(100))
    gender = Column(String(1))
    dob = Column(Date)
    job = Column(String(100))

    # Location details
    street = Column(String(255))
    city = Column(String(100), index=True)
    state = Column(String(2), index=True)
    zip = Column(String(10))
    lat = Column(Float)
    long = Column(Float)
    city_pop = Column(Integer)

    # Fraud label
    is_fraud = Column(Integer, nullable=False, index=True)

    # Additional indexes for common queries
    __table_args__ = (
        Index("idx_fraud_date", "is_fraud", "trans_date_trans_time"),
        Index("idx_merchant_fraud", "merchant", "is_fraud"),
        Index("idx_category_fraud", "category", "is_fraud"),
    )

    def __repr__(self):
        return f"<FraudTransaction(trans_num={self.trans_num}, merchant={self.merchant}, amt={self.amt}, is_fraud={self.is_fraud})>"


class Conversation(Base):
    """Conversation thread model for chat history."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=True)  # Auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title}, messages={len(self.messages)})>"


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)

    # Metadata for assistant messages (renamed to avoid SQLAlchemy reserved word)
    message_metadata = Column(JSON, nullable=True)  # query_type, sql_query, sources, quality_score, etc.

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, conversation_id={self.conversation_id})>"
