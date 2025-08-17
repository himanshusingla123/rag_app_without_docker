# core/__init__.py
"""
Core components for the News RAG System
"""

from .rag_system import NewsRAGSystem
from .database_manager import DatabaseManager

__all__ = [
    'NewsRAGSystem',
    'DatabaseManager'
]