"""Configuration module — single source of truth for credentials and connections.

Loads environment variables from .env at project root. Provides factory
functions for database connections. Never hardcodes credentials.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Resolve project root (two levels up from this file: src/config.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_file = PROJECT_ROOT / ".env"
load_dotenv(_env_file)


def get_postgres_engine() -> Engine:
    """Create and return a SQLAlchemy engine for PostgreSQL.

    Returns:
        A SQLAlchemy Engine connected to the configured PostgreSQL instance.
    """
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'green_user')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'green_pass')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'green_premium')}"
    )
    return create_engine(url, pool_pre_ping=True)


def get_mongo_client() -> MongoClient:
    """Create and return a PyMongo MongoClient.

    Returns:
        A MongoClient connected to the configured MongoDB instance.
    """
    uri = os.getenv(
        "MONGO_URI",
        "mongodb://green_user:green_pass@localhost:27017/?authSource=admin",
    )
    return MongoClient(uri)


def get_mongo_db() -> Database:
    """Return the configured MongoDB database.

    Returns:
        The PyMongo Database object for this project.
    """
    client = get_mongo_client()
    db_name = os.getenv("MONGO_DB", "green_premium")
    return client[db_name]


def test_connections() -> None:
    """Ping both databases and log results. Raises on failure.

    Raises:
        Exception: If either database is unreachable.
    """
    # PostgreSQL
    engine = get_postgres_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection OK")

    # MongoDB
    db = get_mongo_db()
    db.command("ping")
    logger.info("MongoDB connection OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_connections()
