"""
Data models for the News RAG System
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
import json

@dataclass
class NewsArticle:
    """Data structure for news articles"""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    description: Optional[str] = None
    credibility_score: float = 0.0
    bias_score: float = 0.0
    fact_check_score: float = 0.0
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create instance from dictionary"""
        if isinstance(data['published_at'], str):
            data['published_at'] = datetime.fromisoformat(data['published_at'])
        return cls(**data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for vector database storage"""
        return {
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "credibility_score": self.credibility_score,
            "bias_score": self.bias_score,
            "fact_check_score": self.fact_check_score,
            "author": self.author or ""
        }

@dataclass
class FactCheckResult:
    """Fact-checking result structure"""
    claim: str
    verdict: str
    confidence: float
    evidence: List[str]
    sources: List[str]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactCheckResult':
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class BiasAnalysis:
    """Bias analysis results"""
    political_bias: float
    emotional_bias: float
    toxic_bias: float
    readability_complexity: float
    overall_bias_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasAnalysis':
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class SearchResult:
    """Search result structure"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    rank: int
    article_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SystemStatistics:
    """System statistics structure"""
    total_articles: int
    avg_credibility: float
    avg_bias_score: float
    avg_fact_check_score: float
    source_distribution: Dict[str, int]
    latest_update: Optional[str]
    processing_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ProcessingJob:
    """Processing job tracking"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    total_articles: int
    processed_articles: int
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

class NewsArticleEncoder(json.JSONEncoder):
    """Custom JSON encoder for NewsArticle objects"""
    
    def default(self, obj):
        if isinstance(obj, (NewsArticle, FactCheckResult, BiasAnalysis, SearchResult, SystemStatistics)):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)