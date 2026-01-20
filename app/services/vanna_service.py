"""
Vanna AI service for text-to-SQL functionality.
Pattern adapted from property-sales-agentic-assistant.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid slow startup
_vanna_instance = None
_is_trained = False
_executor = ThreadPoolExecutor(max_workers=1)


def _get_vanna_class():
    """Lazy import of Vanna classes."""
    from vanna.chromadb import ChromaDB_VectorStore
    from vanna.openai import OpenAI_Chat

    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    return MyVanna


class VannaService:
    """Service for managing Vanna AI text-to-SQL functionality."""

    def __init__(self):
        """Initialize Vanna service (lazy loading)."""
        self.vanna = None
        self._is_initialized = False

    def _initialize(self):
        """Initialize Vanna instance (called lazily)."""
        if self._is_initialized:
            return

        from app.config import settings

        logger.info("Initializing Vanna AI...")

        MyVanna = _get_vanna_class()
        config = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
            "path": settings.chroma_persist_directory,
        }

        self.vanna = MyVanna(config=config)

        # Connect to PostgreSQL
        self.vanna.connect_to_postgres(
            host=settings.postgres_host,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            port=settings.postgres_port,
        )

        self._is_initialized = True
        logger.info("Vanna AI initialized successfully")

    async def initialize_async(self):
        """Async initialization using thread pool."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._initialize)

    def train(self, ddl: str = None, documentation: str = None, question: str = None, sql: str = None):
        """Train Vanna with DDL, documentation, or Q&A pairs."""
        if not self._is_initialized:
            self._initialize()

        if ddl:
            self.vanna.train(ddl=ddl)
        elif documentation:
            self.vanna.train(documentation=documentation)
        elif question and sql:
            self.vanna.train(question=question, sql=sql)

    async def train_async(self, **kwargs):
        """Async training."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self.train, **kwargs)

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        if not self._is_initialized:
            self._initialize()

        try:
            # Allow LLM to see data for introspection queries (e.g., finding specific merchant names)
            sql = self.vanna.generate_sql(question, allow_llm_to_see_data=True)
            logger.info(f"Generated SQL for '{question}': {sql}")
            return sql
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise

    async def generate_sql_async(self, question: str) -> str:
        """Async SQL generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.generate_sql, question)

    def run_sql(self, sql: str) -> Any:
        """Execute SQL query and return results."""
        if not self._is_initialized:
            self._initialize()

        try:
            result = self.vanna.run_sql(sql)
            logger.info(f"Executed SQL successfully, rows: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise

    async def run_sql_async(self, sql: str) -> Any:
        """Async SQL execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.run_sql, sql)

    def ask(self, question: str, max_retries: int = 2) -> dict:
        """
        Generate and execute SQL for a natural language question.
        Handles intermediate_sql responses by running them and retrying with additional context.
        Returns dict with sql, results, and error (if any).
        """
        try:
            sql = self.generate_sql(question)

            # Handle intermediate_sql responses from Vanna
            # When Vanna is unsure, it returns SQL prefixed with "-- intermediate_sql"
            retry_count = 0
            while sql and sql.strip().startswith("-- intermediate_sql") and retry_count < max_retries:
                logger.info(f"Detected intermediate_sql, running to get context (retry {retry_count + 1})")

                # Remove the comment prefix and run the intermediate query
                intermediate_sql = sql.replace("-- intermediate_sql", "").strip()
                intermediate_results = self.run_sql(intermediate_sql)

                # Convert to list for context
                if hasattr(intermediate_results, 'to_dict'):
                    context_data = intermediate_results.to_dict('records')
                else:
                    context_data = intermediate_results

                # Limit context to avoid token overflow
                context_preview = str(context_data[:20]) if len(context_data) > 20 else str(context_data)

                # Ask again with the additional context
                enhanced_question = f"""{question}

Additional context from database exploration:
Query: {intermediate_sql}
Results: {context_preview}

Please generate the final SQL query to answer the original question using this context."""

                logger.info(f"Retrying with enhanced question containing {len(context_data)} rows of context")
                sql = self.generate_sql(enhanced_question)
                retry_count += 1

            # Clean up any remaining intermediate_sql prefix
            if sql and sql.strip().startswith("-- intermediate_sql"):
                sql = sql.replace("-- intermediate_sql", "").strip()

            results = self.run_sql(sql)

            # Convert DataFrame to dict if needed
            if hasattr(results, 'to_dict'):
                results_dict = results.to_dict('records')
            else:
                results_dict = results

            return {
                "sql": sql,
                "results": results_dict,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            return {
                "sql": None,
                "results": None,
                "error": str(e)
            }

    async def ask_async(self, question: str) -> dict:
        """Async ask."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.ask, question)


# Singleton instance
vanna_service = VannaService()


# Training data for fraud detection
DDL_TRAINING = [
    """
    CREATE TABLE fraud_transactions (
        trans_num VARCHAR(100) PRIMARY KEY,
        trans_date_trans_time TIMESTAMP NOT NULL,
        unix_time BIGINT,
        amt FLOAT NOT NULL,
        cc_num BIGINT NOT NULL,
        merchant VARCHAR(255) NOT NULL,
        category VARCHAR(100) NOT NULL,
        merch_lat FLOAT,
        merch_long FLOAT,
        first VARCHAR(100),
        last VARCHAR(100),
        gender VARCHAR(1),
        dob DATE,
        job VARCHAR(100),
        street VARCHAR(255),
        city VARCHAR(100),
        state VARCHAR(2),
        zip VARCHAR(10),
        lat FLOAT,
        long FLOAT,
        city_pop INTEGER,
        is_fraud INTEGER NOT NULL
    );
    """
]

DOCUMENTATION_TRAINING = [
    "is_fraud: Binary indicator where 0 = legitimate transaction and 1 = fraudulent transaction",
    "category: Transaction category such as gas_transport, grocery_pos, shopping_net, misc_net, etc.",
    "amt: Transaction amount in USD",
    "merchant: Merchant name where the transaction occurred",
    "cc_num: Credit card number (anonymized)",
    "trans_date_trans_time: Timestamp when the transaction occurred",
    "merch_lat, merch_long: Merchant location coordinates",
    "lat, long: Cardholder home location coordinates",
    "city_pop: Population of the cardholder's city",
    "The dataset covers transactions over a two-year period",
    "Use DATE_TRUNC for aggregating by day, month, or year",
    "Calculate fraud rate as: (SUM(is_fraud) * 100.0 / COUNT(*)) for percentage",
    "For searching merchant names, use ILIKE with % wildcards for partial matching since merchant names may contain multiple words",
    "Merchant names are stored as complete strings (e.g., 'fraud_Heller, Gutmann and Zieme'), use ILIKE '%keyword%' for searching",
]

SAMPLE_QUERIES_TRAINING = [
    {
        "question": "How many fraudulent transactions were there last month?",
        "sql": "SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1 AND trans_date_trans_time >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND trans_date_trans_time < DATE_TRUNC('month', CURRENT_DATE)"
    },
    {
        "question": "What is the monthly fraud rate trend?",
        "sql": "SELECT DATE_TRUNC('month', trans_date_trans_time) as month, COUNT(*) as total_transactions, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_percent FROM fraud_transactions GROUP BY month ORDER BY month"
    },
    {
        "question": "Which merchants have the highest fraud rate?",
        "sql": "SELECT merchant, COUNT(*) as total_transactions, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_percent FROM fraud_transactions GROUP BY merchant HAVING COUNT(*) > 100 ORDER BY fraud_rate_percent DESC LIMIT 10"
    },
    {
        "question": "Which categories have the most fraudulent transactions?",
        "sql": "SELECT category, SUM(is_fraud) as fraud_count, COUNT(*) as total_transactions, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_percent FROM fraud_transactions GROUP BY category ORDER BY fraud_count DESC LIMIT 10"
    },
    {
        "question": "What is the average transaction amount for fraud vs legitimate transactions?",
        "sql": "SELECT is_fraud, AVG(amt) as avg_amount, MIN(amt) as min_amount, MAX(amt) as max_amount FROM fraud_transactions GROUP BY is_fraud"
    },
    {
        "question": "Show daily fraud statistics for the last 30 days",
        "sql": "SELECT DATE_TRUNC('day', trans_date_trans_time) as day, COUNT(*) as total, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions WHERE trans_date_trans_time >= CURRENT_DATE - INTERVAL '30 days' GROUP BY day ORDER BY day"
    },
    {
        "question": "Which states have the highest fraud rates?",
        "sql": "SELECT state, COUNT(*) as total, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions GROUP BY state HAVING COUNT(*) > 1000 ORDER BY fraud_rate DESC LIMIT 10"
    },
    {
        "question": "What is the fraud rate by transaction amount range?",
        "sql": "SELECT CASE WHEN amt < 50 THEN '0-50' WHEN amt < 100 THEN '50-100' WHEN amt < 200 THEN '100-200' WHEN amt < 500 THEN '200-500' ELSE '500+' END as amount_range, COUNT(*) as total, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions GROUP BY amount_range ORDER BY amount_range"
    },
    {
        "question": "Show fraud trends by gender",
        "sql": "SELECT gender, COUNT(*) as total, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions GROUP BY gender"
    },
    {
        "question": "What are the top 10 cities with most fraud?",
        "sql": "SELECT city, state, SUM(is_fraud) as fraud_count, COUNT(*) as total, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions GROUP BY city, state ORDER BY fraud_count DESC LIMIT 10"
    },
    {
        "question": "Show me transactions from merchant Heller",
        "sql": "SELECT * FROM fraud_transactions WHERE merchant ILIKE '%Heller%' LIMIT 100"
    },
    {
        "question": "Tell me about merchant fraud_Smith and Sons data",
        "sql": "SELECT COUNT(*) as total_transactions, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate, AVG(amt) as avg_amount FROM fraud_transactions WHERE merchant ILIKE '%Smith%'"
    },
    {
        "question": "What is the fraud rate for merchant Johnson LLC?",
        "sql": "SELECT merchant, COUNT(*) as total, SUM(is_fraud) as fraud_count, ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate FROM fraud_transactions WHERE merchant ILIKE '%Johnson%' GROUP BY merchant"
    }
]


async def train_vanna_model():
    """Train Vanna model with fraud detection knowledge."""
    global _is_trained

    if _is_trained:
        logger.info("Vanna model already trained, skipping...")
        return

    logger.info("Training Vanna AI model...")

    # Initialize first
    await vanna_service.initialize_async()

    # Train with DDL
    logger.info("Training with schema (DDL)...")
    for ddl in DDL_TRAINING:
        vanna_service.train(ddl=ddl)

    # Train with documentation
    logger.info("Training with documentation...")
    for doc in DOCUMENTATION_TRAINING:
        vanna_service.train(documentation=doc)

    # Train with sample Q&A
    logger.info("Training with sample queries...")
    for sample in SAMPLE_QUERIES_TRAINING:
        vanna_service.train(question=sample["question"], sql=sample["sql"])

    _is_trained = True
    logger.info("Vanna AI training completed!")
