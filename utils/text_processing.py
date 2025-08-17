"""
Text processing utilities for the News RAG System
"""
import re
import hashlib
import nltk
from typing import List, Dict
from textstat import flesch_kincaid_grade, automated_readability_index

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
        
        return text.strip()
    
    @staticmethod
    def generate_article_id(title: str, url: str) -> str:
        """Generate unique ID for article"""
        return hashlib.md5(f"{url}{title}".encode()).hexdigest()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            sentences = nltk.sent_tokenize(text)
            return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
        except Exception:
            # Fallback to simple sentence splitting
            sentences = text.split('.')
            return [sentence.strip() + '.' for sentence in sentences if len(sentence.strip()) > 10]
    
    @staticmethod
    def calculate_readability_scores(text: str) -> Dict[str, float]:
        """Calculate readability scores"""
        scores = {}
        
        try:
            scores['flesch_kincaid'] = flesch_kincaid_grade(text)
        except Exception:
            scores['flesch_kincaid'] = 0.0
        
        try:
            scores['automated_readability'] = automated_readability_index(text)
        except Exception:
            scores['automated_readability'] = 0.0
        
        # Calculate complexity score (0-1)
        if scores['flesch_kincaid'] > 0 and scores['automated_readability'] > 0:
            scores['complexity'] = min(1.0, (scores['flesch_kincaid'] + scores['automated_readability']) / 20.0)
        else:
            scores['complexity'] = 0.0
        
        return scores
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'said', 'says', 'news'
        }
        
        # Filter out stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        if add_ellipsis:
            # Try to break at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # If we can find a space reasonably close to the end
                truncated = truncated[:last_space]
            truncated += "..."
        
        return truncated
    
    @staticmethod
    def count_emotional_indicators(text: str) -> Dict[str, int]:
        """Count emotional indicators in text"""
        indicators = {
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        return indicators
    
    @staticmethod
    def extract_numbers_and_stats(text: str) -> List[str]:
        """Extract numbers, percentages, and statistics from text"""
        patterns = [
            r'\d+%',  # Percentages
            r'\d+\s*percent',  # Percent written out
            r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?',  # Numbers with commas, decimals, units
            r'\d+(?:\.\d+)?\s*(?:times|fold|x)',  # Multipliers
        ]
        
        stats = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats.extend(matches)
        
        return stats[:10]  # Limit to first 10 matches