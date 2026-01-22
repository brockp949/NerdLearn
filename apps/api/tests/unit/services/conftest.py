"""
Service unit test conftest - sets environment variables BEFORE app imports.
This file is processed BEFORE test files are collected.
"""

import os
import sys


def pytest_configure(config):
    """Set environment variables before test collection."""
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "test"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["SECRET_KEY"] = "test-secret"
    os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000"]'


# Also set at module level for any imports that happen during collection
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "test"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000"]'
