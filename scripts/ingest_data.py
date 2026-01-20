"""
Data ingestion script to load fraudTrain.csv and fraudTest.csv into PostgreSQL.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import create_engine, text
from app.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_csv_to_db(csv_path: str, table_name: str = "fraud_transactions"):
    """Load CSV file into PostgreSQL database."""

    logger.info(f"Loading data from {csv_path}...")

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Data preprocessing
    logger.info("Preprocessing data...")

    # Convert datetime columns
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    # Handle potential null values
    df = df.where(pd.notnull(df), None)

    # Create database engine
    engine = create_engine(settings.sync_database_url)

    # Load to database (append mode to support both train and test)
    logger.info(f"Inserting data into {table_name}...")
    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        chunksize=10000,
        method='multi'
    )

    logger.info(f"Successfully loaded {len(df)} rows into {table_name}")

    return len(df)


def main():
    """Main ingestion function."""

    # Check if CSV files exist
    data_dir = Path("data")
    train_csv = data_dir / "fraudTrain.csv"
    test_csv = data_dir / "fraudTest.csv"

    if not train_csv.exists():
        logger.error(f"fraudTrain.csv not found at {train_csv}")
        logger.info("Please place fraudTrain.csv in the data/ directory")
        return

    if not test_csv.exists():
        logger.error(f"fraudTest.csv not found at {test_csv}")
        logger.info("Please place fraudTest.csv in the data/ directory")
        return

    # Create engine and drop existing table
    engine = create_engine(settings.sync_database_url)
    logger.info("Dropping existing fraud_transactions table if exists...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fraud_transactions CASCADE"))
        conn.commit()

    total_rows = 0

    # Load training data
    logger.info("=" * 60)
    logger.info("Loading fraudTrain.csv...")
    logger.info("=" * 60)
    rows = load_csv_to_db(str(train_csv))
    total_rows += rows

    # Load test data
    logger.info("=" * 60)
    logger.info("Loading fraudTest.csv...")
    logger.info("=" * 60)
    rows = load_csv_to_db(str(test_csv))
    total_rows += rows

    logger.info("=" * 60)
    logger.info(f"Data ingestion complete! Total rows: {total_rows}")
    logger.info("=" * 60)

    # Create indexes
    logger.info("Creating indexes...")
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fraud ON fraud_transactions(is_fraud)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_merchant ON fraud_transactions(merchant)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_category ON fraud_transactions(category)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trans_date ON fraud_transactions(trans_date_trans_time)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fraud_date ON fraud_transactions(is_fraud, trans_date_trans_time)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_merchant_fraud ON fraud_transactions(merchant, is_fraud)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_category_fraud ON fraud_transactions(category, is_fraud)"))
        conn.commit()

    logger.info("Indexes created successfully!")

    # Show some statistics
    logger.info("\nDatabase Statistics:")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM fraud_transactions"))
        total = result.fetchone()[0]
        logger.info(f"  Total transactions: {total}")

        result = conn.execute(text("SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1"))
        fraud = result.fetchone()[0]
        logger.info(f"  Fraudulent transactions: {fraud}")
        logger.info(f"  Fraud rate: {fraud/total*100:.2f}%")

        result = conn.execute(text("SELECT MIN(trans_date_trans_time), MAX(trans_date_trans_time) FROM fraud_transactions"))
        min_date, max_date = result.fetchone()
        logger.info(f"  Date range: {min_date} to {max_date}")


if __name__ == "__main__":
    main()
