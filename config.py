"""
Configuration settings for News RAG System
"""
# import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "news_db"

# Database configuration
SQLITE_DB_PATH = DATA_DIR / "news_metadata.db"
CHROMADB_PATH = str(DB_DIR)
COLLECTION_NAME = "news_articles"

# RSS Feed sources with credibility ratings
RSS_FEEDS = {
    'BBC News': {
        'url': 'http://feeds.bbci.co.uk/news/rss.xml',
        'credibility': 0.9
    },
    'Reuters': {
        'url': 'https://feeds.reuters.com/reuters/topNews',
        'credibility': 0.95
    },
    'AP News': {
        'url': 'https://feeds.apnews.com/rss/apf-topnews',
        'credibility': 0.95
    },
    'NPR': {
        'url': 'https://feeds.npr.org/1001/rss.xml',
        'credibility': 0.85
    },
    'CNN': {
        'url': 'http://rss.cnn.com/rss/edition.rss',
        'credibility': 0.75
    },
    'Guardian': {
        'url': 'https://www.theguardian.com/world/rss',
        'credibility': 0.8
    },
    'Washington Post': {
        'url': 'https://feeds.washingtonpost.com/rss/world',
        'credibility': 0.8
    },
    'Al Jazeera': {
        'url': 'https://www.aljazeera.com/xml/rss/all.xml',
        'credibility': 0.7
    }
}

# Model configurations
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
BIAS_MODEL_NAME = "unitary/toxic-bert"
DEVICE = 'cpu'  # Use 'cuda' if GPU available

# Processing limits
MAX_ARTICLES_PER_SOURCE = 20
MAX_CONTENT_LENGTH = 2000
MAX_CLAIMS_PER_ARTICLE = 5

# Bias detection keywords
POLITICAL_KEYWORDS = {
    'left': ['progressive', 'liberal', 'democrat', 'socialism'],
    'right': ['conservative', 'republican', 'capitalism', 'traditional']
}

# Fact-checking patterns
FACT_CHECK_INDICATORS = [
    'according to', 'reported that', 'studies show', 'data reveals',
    'statistics indicate', 'research suggests', 'experts say'
]

MISINFORMATION_PATTERNS = [
    r'100% proven', r'doctors hate this', r'secret that [\w\s]+ don\'t want',
    r'miracle cure', r'shocking truth', r'they don\'t want you to know'
]

# Similarity thresholds
HIGH_SIMILARITY_THRESHOLD = 0.3
MODERATE_SIMILARITY_THRESHOLD = 0.6

# UI Configuration
PAGE_TITLE = "News RAG System with Fact-Checking"
PAGE_ICON = "📰"
DEFAULT_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS = 20

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)