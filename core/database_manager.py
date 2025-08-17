"""
Database manager for News RAG System
Handles both SQLite and ChromaDB operations
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
import chromadb
# from chromadb.config import Settings
import json

from models.data_models import NewsArticle, SearchResult, SystemStatistics
from config import SQLITE_DB_PATH, CHROMADB_PATH, COLLECTION_NAME


class DatabaseManager:
    """Manages database operations for the News RAG System"""
    
    def __init__(self):
        self.sqlite_db_path = SQLITE_DB_PATH
        self.chromadb_path = CHROMADB_PATH
        self.collection_name = COLLECTION_NAME
        
        # Initialize databases
        self._init_sqlite()
        self._init_chromadb()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        # Create articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                published_at TIMESTAMP NOT NULL,
                author TEXT,
                credibility_score REAL DEFAULT 0.0,
                bias_score REAL DEFAULT 0.0,
                fact_check_score REAL DEFAULT 0.0,
                content_length INTEGER DEFAULT 0,
                keywords TEXT,  -- JSON string of keywords
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create processing jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total_articles INTEGER DEFAULT 0,
                processed_articles INTEGER DEFAULT 0,
                errors TEXT,  -- JSON string of errors
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create fact checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fact_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT NOT NULL,
                claim TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                evidence TEXT,  -- JSON string of evidence
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_source ON articles (source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_published ON articles (published_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_credibility ON articles (credibility_score)')
        
        conn.commit()
        conn.close()
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path=self.chromadb_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_article(self, article: NewsArticle) -> bool:
        """Store article in both SQLite and ChromaDB"""
        try:
            # Store in SQLite
            self._store_article_metadata(article)
            
            # Store in ChromaDB if embedding exists
            if article.embedding:
                self._store_article_vector(article)
            
            return True
        except Exception as e:
            print(f"Error storing article {article.id}: {str(e)}")
            return False
    
    def _store_article_metadata(self, article: NewsArticle):
        """Store article metadata in SQLite"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        # Convert keywords to JSON string if they exist
        keywords_json = json.dumps(getattr(article, 'keywords', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO articles 
            (id, title, source, url, published_at, author, credibility_score, 
             bias_score, fact_check_score, content_length, keywords, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article.id, article.title, article.source, article.url,
            article.published_at, article.author, article.credibility_score,
            article.bias_score, article.fact_check_score, 
            len(article.content), keywords_json, datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_article_vector(self, article: NewsArticle):
        """Store article vector in ChromaDB"""
        try:
            # Check if article already exists
            existing = self.collection.get(ids=[article.id])
            
            if existing['ids']:
                # Update existing
                self.collection.update(
                    ids=[article.id],
                    embeddings=[article.embedding],
                    documents=[f"{article.title}\n\n{article.content}"],
                    metadatas=[article.get_metadata()]
                )
            else:
                # Add new
                self.collection.add(
                    ids=[article.id],
                    embeddings=[article.embedding],
                    documents=[f"{article.title}\n\n{article.content}"],
                    metadatas=[article.get_metadata()]
                )
        except Exception as e:
            print(f"Error storing vector for article {article.id}: {str(e)}")
    
    def search_articles(self, query_embedding: List[float], n_results: int = 10) -> List[SearchResult]:
        """Search articles using vector similarity"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            search_results = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_result = SearchResult(
                        content=doc,
                        metadata=metadata,
                        relevance_score=1 - distance,
                        rank=i + 1,
                        article_id=results['ids'][0][i] if results['ids'] else None
                    )
                    search_results.append(search_result)
            
            return search_results
        except Exception as e:
            print(f"Error searching articles: {str(e)}")
            return []
    
    def get_article_by_id(self, article_id: str) -> Optional[NewsArticle]:
        """Get article by ID from SQLite"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_article(row)
        return None
    
    def get_articles_by_source(self, source: str, limit: int = 10) -> List[NewsArticle]:
        """Get articles by source"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM articles WHERE source = ? ORDER BY published_at DESC LIMIT ?',
            (source, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_article(row) for row in rows]
    
    def get_recent_articles(self, limit: int = 50) -> List[NewsArticle]:
        """Get most recent articles"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM articles ORDER BY published_at DESC LIMIT ?',
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_article(row) for row in rows]
    
    def get_statistics(self) -> SystemStatistics:
        """Get system statistics"""
        conn = sqlite3.connect(self.sqlite_db_path)
        
        try:
            df = pd.read_sql_query("SELECT * FROM articles", conn)
            
            if df.empty:
                return SystemStatistics(
                    total_articles=0,
                    avg_credibility=0.0,
                    avg_bias_score=0.0,
                    avg_fact_check_score=0.0,
                    source_distribution={},
                    latest_update=None
                )
            
            stats = SystemStatistics(
                total_articles=len(df),
                avg_credibility=df['credibility_score'].mean(),
                avg_bias_score=df['bias_score'].mean(),
                avg_fact_check_score=df['fact_check_score'].mean(),
                source_distribution=df['source'].value_counts().to_dict(),
                latest_update=df['created_at'].max() if 'created_at' in df.columns else None
            )
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return SystemStatistics(
                total_articles=0,
                avg_credibility=0.0,
                avg_bias_score=0.0,
                avg_fact_check_score=0.0,
                source_distribution={},
                latest_update=None
            )
        finally:
            conn.close()
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from both databases"""
        try:
            # Delete from SQLite
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM articles WHERE id = ?', (article_id,))
            cursor.execute('DELETE FROM fact_checks WHERE article_id = ?', (article_id,))
            conn.commit()
            conn.close()
            
            # Delete from ChromaDB
            try:
                self.collection.delete(ids=[article_id])
            except Exception:
                pass  # Article might not exist in vector database
            
            return True
        except Exception as e:
            print(f"Error deleting article {article_id}: {str(e)}")
            return False
    
    def cleanup_old_articles(self, days: int = 30) -> int:
        """Remove articles older than specified days"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        # Get IDs of old articles
        cursor.execute('''
            SELECT id FROM articles 
            WHERE published_at < date('now', '-{} days')
        '''.format(days))
        old_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete from SQLite
        cursor.execute('''
            DELETE FROM articles 
            WHERE published_at < date('now', '-{} days')
        '''.format(days))
        
        cursor.execute('''
            DELETE FROM fact_checks 
            WHERE article_id NOT IN (SELECT id FROM articles)
        ''')
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Delete from ChromaDB
        if old_ids:
            try:
                self.collection.delete(ids=old_ids)
            except Exception:
                pass
        
        return deleted_count
    
    def _row_to_article(self, row) -> NewsArticle:
        """Convert SQLite row to NewsArticle object"""
        # SQLite row structure based on our table schema
        return NewsArticle(
            id=row[0],
            title=row[1],
            source=row[2],
            url=row[3],
            published_at=datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4],
            author=row[5],
            credibility_score=row[6] or 0.0,
            bias_score=row[7] or 0.0,
            fact_check_score=row[8] or 0.0,
            content="",  # Content not stored in SQLite, only in ChromaDB
            description=""
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection information"""
        try:
            return {
                'name': self.collection.name,
                'count': self.collection.count(),
                'metadata': self.collection.metadata
            }
        except Exception as e:
            return {'error': str(e)}