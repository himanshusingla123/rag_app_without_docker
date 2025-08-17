"""
Bias detection engine for analyzing news articles
"""
import streamlit as st
from typing import Dict, List
from transformers import pipeline
import numpy as np

from models.data_models import BiasAnalysis
from utils.text_processing import TextProcessor
from config import BIAS_MODEL_NAME, POLITICAL_KEYWORDS, DEVICE


class BiasDetectionEngine:
    """Detect various types of bias in news articles using transformer models"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.political_keywords = POLITICAL_KEYWORDS
        
        # Load bias detection model (free)
        self.bias_classifier = self._load_bias_model()
        
        # Sentiment analysis pipeline
        self.sentiment_analyzer = self._load_sentiment_model()
        
        # Emotional language indicators
        self.emotional_indicators = {
            'sensational': ['shocking', 'amazing', 'incredible', 'unbelievable', 'devastating', 'outrageous'],
            'fear': ['dangerous', 'threat', 'crisis', 'disaster', 'panic', 'terror', 'alarming'],
            'anger': ['furious', 'outraged', 'disgusting', 'appalling', 'scandalous', 'betrayal'],
            'positive': ['wonderful', 'fantastic', 'excellent', 'perfect', 'brilliant', 'outstanding']
        }
    
    def _load_bias_model(self):
        """Load bias detection model"""
        try:
            return pipeline(
                "text-classification",
                model=BIAS_MODEL_NAME,
                tokenizer=BIAS_MODEL_NAME,
                device=-1 if DEVICE == 'cpu' else 0
            )
        except Exception as e:
            st.warning(f"Bias detection model not available: {str(e)}")
            return None
    
    def _load_sentiment_model(self):
        """Load sentiment analysis model"""
        try:
            return pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1 if DEVICE == 'cpu' else 0
            )
        except Exception:
            # Fallback to a simpler model
            try:
                return pipeline("sentiment-analysis", device=-1 if DEVICE == 'cpu' else 0)
            except Exception:
                return None
    
    def analyze_bias(self, text: str, title: str = "") -> BiasAnalysis:
        """Comprehensive bias analysis of text"""
        if not text:
            return BiasAnalysis(
                political_bias=0.0,
                emotional_bias=0.0,
                toxic_bias=0.0,
                readability_complexity=0.0,
                overall_bias_score=0.0
            )
        
        # Combine title and text for analysis
        full_text = f"{title} {text}" if title else text
        
        # Analyze different types of bias
        political_bias = self._detect_political_bias(full_text)
        emotional_bias = self._detect_emotional_bias(full_text)
        toxic_bias = self._detect_toxic_bias(full_text)
        readability_complexity = self._analyze_readability_complexity(full_text)
        
        # Calculate overall bias score
        overall_bias_score = self._calculate_overall_bias_score(
            political_bias, emotional_bias, toxic_bias, readability_complexity
        )
        
        return BiasAnalysis(
            political_bias=political_bias,
            emotional_bias=emotional_bias,
            toxic_bias=toxic_bias,
            readability_complexity=readability_complexity,
            overall_bias_score=overall_bias_score
        )
    
    def _detect_political_bias(self, text: str) -> float:
        """Detect political bias using keyword analysis and context"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count political keywords
        left_count = sum(text_lower.count(word) for word in self.political_keywords['left'])
        right_count = sum(text_lower.count(word) for word in self.political_keywords['right'])
        
        total_political = left_count + right_count
        
        if total_political == 0:
            return 0.0
        
        # Calculate bias score (0 = neutral, 1 = highly biased)
        bias_ratio = abs(left_count - right_count) / total_political
        
        # Consider frequency relative to text length
        frequency_factor = min(1.0, total_political / (len(text.split()) * 0.01))
        
        return min(1.0, bias_ratio * frequency_factor)
    
    def _detect_emotional_bias(self, text: str) -> float:
        """Detect emotional bias through linguistic analysis"""
        if not text:
            return 0.0
        
        emotional_indicators = self.text_processor.count_emotional_indicators(text)
        text_lower = text.lower()
        
        # Count emotional language
        emotional_word_count = 0
        for category, words in self.emotional_indicators.items():
            emotional_word_count += sum(text_lower.count(word) for word in words)
        
        # Calculate emotional bias factors
        exclamation_factor = min(1.0, emotional_indicators['exclamations'] * 0.1)
        caps_factor = min(1.0, emotional_indicators['caps_ratio'] * 2.0)
        emotional_words_factor = min(1.0, emotional_word_count / (len(text.split()) * 0.1))
        
        # Sentiment analysis if model available
        sentiment_factor = 0.0
        if self.sentiment_analyzer:
            try:
                # Analyze chunks of text due to model limitations
                sentiment_scores = []
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for chunk in chunks[:5]:  # Limit to first 5 chunks
                    result = self.sentiment_analyzer(chunk)
                    if result and len(result) > 0:
                        score = result[0].get('score', 0)
                        label = result[0].get('label', '').upper()
                        
                        # Convert to bias score (extreme sentiment = higher bias)
                        if label in ['NEGATIVE', 'POSITIVE']:
                            sentiment_scores.append(score)
                
                if sentiment_scores:
                    # High confidence in extreme sentiment indicates bias
                    avg_sentiment = np.mean(sentiment_scores)
                    sentiment_factor = min(1.0, avg_sentiment * 0.5)
                    
            except Exception:
                pass
        
        # Combine all emotional bias factors
        emotional_bias = np.mean([
            exclamation_factor,
            caps_factor,
            emotional_words_factor,
            sentiment_factor
        ])
        
        return min(1.0, emotional_bias)
    
    def _detect_toxic_bias(self, text: str) -> float:
        """Detect toxic language and bias"""
        if not text or not self.bias_classifier:
            return 0.0
        
        try:
            # Analyze text in chunks due to model limitations
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            toxic_scores = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                result = self.bias_classifier(chunk)
                
                if isinstance(result, list) and len(result) > 0:
                    score = result[0].get('score', 0.0)
                    label = result[0].get('label', '').upper()
                    
                    if label == 'TOXIC':
                        toxic_scores.append(score)
                    elif label == 'NOT_TOXIC':
                        toxic_scores.append(1.0 - score)  # Invert score
            
            return np.mean(toxic_scores) if toxic_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_readability_complexity(self, text: str) -> float:
        """Analyze readability as a bias indicator"""
        if not text:
            return 0.0
        
        try:
            readability_scores = self.text_processor.calculate_readability_scores(text)
            return readability_scores.get('complexity', 0.0)
        except Exception:
            return 0.0
    
    def _calculate_overall_bias_score(self, political: float, emotional: float, 
                                    toxic: float, complexity: float) -> float:
        """Calculate overall bias score from individual components"""
        # Weight different types of bias
        weights = {
            'political': 0.3,
            'emotional': 0.3,
            'toxic': 0.3,
            'complexity': 0.1
        }
        
        overall_score = (
            political * weights['political'] +
            emotional * weights['emotional'] +
            toxic * weights['toxic'] +
            complexity * weights['complexity']
        )
        
        return min(1.0, overall_score)
    
    def get_bias_explanation(self, bias_analysis: BiasAnalysis) -> Dict[str, str]:
        """Generate human-readable explanations for bias scores"""
        explanations = {}
        
        # Political bias explanation
        if bias_analysis.political_bias < 0.3:
            explanations['political'] = "Low political bias detected"
        elif bias_analysis.political_bias < 0.6:
            explanations['political'] = "Moderate political bias detected"
        else:
            explanations['political'] = "High political bias detected"
        
        # Emotional bias explanation
        if bias_analysis.emotional_bias < 0.3:
            explanations['emotional'] = "Neutral emotional tone"
        elif bias_analysis.emotional_bias < 0.6:
            explanations['emotional'] = "Somewhat emotional language"
        else:
            explanations['emotional'] = "Highly emotional/sensational language"
        
        # Toxic bias explanation
        if bias_analysis.toxic_bias < 0.2:
            explanations['toxic'] = "No toxic language detected"
        elif bias_analysis.toxic_bias < 0.5:
            explanations['toxic'] = "Some potentially problematic language"
        else:
            explanations['toxic'] = "Toxic or offensive language detected"
        
        # Readability explanation
        if bias_analysis.readability_complexity < 0.3:
            explanations['readability'] = "Easy to read"
        elif bias_analysis.readability_complexity < 0.6:
            explanations['readability'] = "Moderate complexity"
        else:
            explanations['readability'] = "Complex/academic language"
        
        # Overall assessment
        if bias_analysis.overall_bias_score < 0.3:
            explanations['overall'] = "Generally unbiased reporting"
        elif bias_analysis.overall_bias_score < 0.6:
            explanations['overall'] = "Some bias indicators present"
        else:
            explanations['overall'] = "Significant bias detected"
        
        return explanations
    
    def detect_bias_patterns(self, articles: List[str]) -> Dict[str, float]:
        """Detect bias patterns across multiple articles"""
        if not articles:
            return {}
        
        bias_analyses = [self.analyze_bias(article) for article in articles]
        
        patterns = {
            'avg_political_bias': np.mean([b.political_bias for b in bias_analyses]),
            'avg_emotional_bias': np.mean([b.emotional_bias for b in bias_analyses]),
            'avg_toxic_bias': np.mean([b.toxic_bias for b in bias_analyses]),
            'consistency_score': 1.0 - np.std([b.overall_bias_score for b in bias_analyses]),
            'high_bias_articles': sum(1 for b in bias_analyses if b.overall_bias_score > 0.6)
        }
        
        return patterns
    
    def compare_sources(self, source_articles: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Compare bias patterns between different news sources"""
        source_bias = {}
        
        for source, articles in source_articles.items():
            if articles:
                patterns = self.detect_bias_patterns(articles)
                source_bias[source] = patterns
        
        return source_bias