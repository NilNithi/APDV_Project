"""Smoke tests for src/config.py — verifies env loading without live DBs."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_mongo_db, get_postgres_engine, test_connections


def test_postgres_engine_url_uses_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine URL should incorporate env vars when present."""
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_HOST", "testhost")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

    # Re-import to pick up monkeypatched env (config uses os.getenv at call time)
    engine = get_postgres_engine()
    url = str(engine.url)
    assert "testuser" in url
    assert "testhost" in url
    assert "testdb" in url


def test_mongo_db_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_mongo_db should use MONGO_DB env var for database name."""
    monkeypatch.setenv("MONGO_DB", "my_test_db")
    monkeypatch.setenv(
        "MONGO_URI", "mongodb://user:pass@localhost:27017/?authSource=admin"
    )
    with patch("src.config.MongoClient") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=MagicMock())
        db = get_mongo_db()
        # Verify the correct DB name was used
        mock_client.return_value.__getitem__.assert_called_with("my_test_db")


def test_test_connections_succeeds_with_mocked_dbs() -> None:
    """test_connections should not raise when both DBs respond OK."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_db = MagicMock()
    mock_db.command.return_value = {"ok": 1}

    with (
        patch("src.config.get_postgres_engine", return_value=mock_engine),
        patch("src.config.get_mongo_db", return_value=mock_db),
    ):
        test_connections()  # should not raise
