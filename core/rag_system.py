"""
Main RAG system orchestrating all components
"""
import streamlit as st
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from models.data_models import NewsArticle, SearchResult, SystemStatistics, ProcessingJob
from engines.ingestion_engine import NewsIngestionEngine
from engines.bias_detection_engine import BiasDetectionEngine
from engines.fact_checking_engine import FactCheckingEngine
from engines.embedding_engine import EmbeddingEngine
from core.database_manager import DatabaseManager
from utils.text_processing import TextProcessor
# from config import MAX_ARTICLES_PER_SOURCE


class NewsRAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self):
        # Initialize all components
        self.ingestion_engine = NewsIngestionEngine()
        self.bias_detector = BiasDetectionEngine()
        self.embedding_engine = EmbeddingEngine()
        self.database_manager = DatabaseManager()
        self.text_processor = TextProcessor()
        
        # Initialize fact checker with database and embedding model
        self.fact_checker = FactCheckingEngine(
            self.database_manager.collection,
            self.embedding_engine
        )
        
        # System state
        self._initialized = True
    
    def initialize_system(self) -> bool:
        """Initialize the complete system"""
        try:
            # Check if all components are properly initialized
            components_status = {
                'ingestion_engine': self.ingestion_engine is not None,
                'bias_detector': self.bias_detector is not None,
                'embedding_engine': self.embedding_engine.model is not None,
                'database_manager': self.database_manager is not None,
                'fact_checker': self.fact_checker is not None
            }
            
            failed_components = [name for name, status in components_status.items() if not status]
            
            if failed_components:
                st.error(f"Failed to initialize components: {', '.join(failed_components)}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return False
    
    def process_news_batch(self, max_articles: int = 50) -> ProcessingJob:
        """Process a batch of news articles"""
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = ProcessingJob(
            job_id=job_id,
            status='processing',
            total_articles=0,
            processed_articles=0,
            errors=[],
            started_at=datetime.now()
        )
        
        try:
            # Fetch articles
            st.info("Fetching articles from RSS feeds...")
            articles = self.ingestion_engine.fetch_articles_from_rss(max_articles)
            job.total_articles = len(articles)
            
            if not articles:
                job.status = 'completed'
                job.completed_at = datetime.now()
                return job
            
            # Process articles
            processed_articles = self.process_articles(articles, job)
            
            job.processed_articles = len(processed_articles)
            job.status = 'completed'
            job.completed_at = datetime.now()
            
            return job
            
        except Exception as e:
            job.status = 'failed'
            job.errors.append(str(e))
            job.completed_at = datetime.now()
            return job
    
    def process_articles(self, articles: List[NewsArticle], job: ProcessingJob = None) -> List[NewsArticle]:
        """Process articles through the complete pipeline"""
        processed_articles = []
        
        if not articles:
            return processed_articles
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, article in enumerate(articles):
            try:
                status_text.text(f"Processing: {article.source} - {article.title[:50]}...")
                
                # Step 1: Generate embeddings
                article.embedding = self.embedding_engine.encode_article(article)
                
                if article.embedding is None:
                    if job:
                        job.errors.append(f"Failed to generate embedding for: {article.title}")
                    continue
                
                # Step 2: Detect bias
                bias_analysis = self.bias_detector.analyze_bias(article.content, article.title)
                article.bias_score = bias_analysis.overall_bias_score
                
                # Step 3: Fact-check
                fact_check_results = self.fact_checker.fact_check_article(article)
                
                # Calculate fact-check score
                if fact_check_results:
                    confidences = [r.confidence for r in fact_check_results]
                    verdicts = [r.verdict for r in fact_check_results]
                    
                    # Weight different verdicts
                    verdict_weights = {
                        'Likely True': 1.0,
                        'Partially Supported': 0.7,
                        'Insufficient Evidence': 0.5,
                        'Questionable': 0.3,
                        'Likely False': 0.1
                    }
                    
                    weighted_scores = []
                    for verdict, confidence in zip(verdicts, confidences):
                        weight = verdict_weights.get(verdict, 0.5)
                        weighted_scores.append(weight * confidence)
                    
                    article.fact_check_score = np.mean(weighted_scores) if weighted_scores else 0.5
                else:
                    # Base score on misinformation patterns
                    misinfo_score = self.fact_checker.check_misinformation_patterns(article.content)
                    article.fact_check_score = max(0.1, 1.0 - misinfo_score)
                
                # Step 4: Store in database
                if self.database_manager.store_article(article):
                    processed_articles.append(article)
                else:
                    if job:
                        job.errors.append(f"Failed to store article: {article.title}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(articles))
                
            except Exception as e:
                error_msg = f"Error processing article {article.title}: {str(e)}"
                if job:
                    job.errors.append(error_msg)
                st.warning(error_msg)
                continue
        
        status_text.text("Processing completed!")
        return processed_articles
    
    def search_articles(self, query: str, n_results: int = 10, 
                       filters: Dict = None) -> List[SearchResult]:
        """Search articles using RAG"""
        if not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_engine.encode_text(query)
            
            if query_embedding is None:
                st.error("Failed to generate query embedding")
                return []
            
            # Search in vector database
            search_results = self.database_manager.search_articles(
                query_embedding.tolist(), 
                n_results
            )
            
            # Apply filters if provided
            if filters:
                search_results = self._apply_search_filters(search_results, filters)
            
            return search_results
            
        except Exception as e:
            st.error(f"Error searching articles: {str(e)}")
            return []
    
    def _apply_search_filters(self, results: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result.metadata
            
            # Filter by source
            if filters.get('sources') and metadata.get('source') not in filters['sources']:
                continue
            
            # Filter by credibility score
            min_credibility = filters.get('min_credibility', 0.0)
            if metadata.get('credibility_score', 0.0) < min_credibility:
                continue
            
            # Filter by bias score
            max_bias = filters.get('max_bias', 1.0)
            if metadata.get('bias_score', 0.0) > max_bias:
                continue
            
            # Filter by fact-check score
            min_fact_check = filters.get('min_fact_check', 0.0)
            if metadata.get('fact_check_score', 0.0) < min_fact_check:
                continue
            
            # Filter by date range
            if filters.get('date_range'):
                published_at = metadata.get('published_at')
                if published_at:
                    # Add date filtering logic here
                    pass
            
            filtered_results.append(result)
        
        return filtered_results
    
    def get_article_analysis(self, article_id: str) -> Dict:
        """Get comprehensive analysis for a specific article"""
        try:
            article = self.database_manager.get_article_by_id(article_id)
            
            if not article:
                return {'error': 'Article not found'}
            
            # Get full content from vector database if needed
            vector_results = self.database_manager.collection.get(ids=[article_id])
            full_content = ""
            
            if vector_results['documents'] and len(vector_results['documents']) > 0:
                full_content = vector_results['documents'][0]
            
            # Perform comprehensive analysis
            bias_analysis = self.bias_detector.analyze_bias(full_content, article.title)
            fact_check_results = self.fact_checker.fact_check_article(article)
            source_credibility = self.fact_checker.analyze_source_credibility(article)
            
            # Extract keywords and statistics
            keywords = self.text_processor.extract_keywords(full_content)
            stats = self.text_processor.extract_numbers_and_stats(full_content)
            readability = self.text_processor.calculate_readability_scores(full_content)
            
            return {
                'article': article.to_dict(),
                'bias_analysis': bias_analysis.to_dict(),
                'fact_check_results': [r.to_dict() for r in fact_check_results],
                'source_credibility': source_credibility,
                'keywords': keywords,
                'statistics': stats,
                'readability': readability,
                'content_length': len(full_content),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_trending_topics(self, days: int = 7, top_k: int = 10) -> List[Dict]:
        """Get trending topics based on recent articles"""
        try:
            # Get recent articles
            recent_articles = self.database_manager.get_recent_articles(limit=100)
            
            if not recent_articles:
                return []
            
            # Extract keywords from all recent articles
            all_keywords = []
            for article in recent_articles:
                # Get full content from vector database
                vector_results = self.database_manager.collection.get(ids=[article.id])
                if vector_results['documents'] and len(vector_results['documents']) > 0:
                    content = vector_results['documents'][0]
                    keywords = self.text_processor.extract_keywords(content, max_keywords=5)
                    all_keywords.extend(keywords)
            
            # Count keyword frequency
            keyword_freq = {}
            for keyword in all_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            # Sort and return trending topics
            trending = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {'topic': topic, 'frequency': freq, 'trend_score': freq / len(recent_articles)}
                for topic, freq in trending[:top_k]
            ]
            
        except Exception as e:
            st.error(f"Error getting trending topics: {str(e)}")
            return []
    
    def compare_sources(self, topic: str = "", days: int = 7) -> Dict:
        """Compare bias and credibility across news sources"""
        try:
            if topic:
                # Search for articles on the topic
                search_results = self.search_articles(topic, n_results=50)
                article_ids = [r.article_id for r in search_results if r.article_id]
                articles = [self.database_manager.get_article_by_id(aid) for aid in article_ids]
                articles = [a for a in articles if a is not None]
            else:
                # Use recent articles
                articles = self.database_manager.get_recent_articles(limit=100)
            
            if not articles:
                return {}
            
            # Group by source
            source_data = {}
            for article in articles:
                source = article.source
                if source not in source_data:
                    source_data[source] = {
                        'articles': [],
                        'credibility_scores': [],
                        'bias_scores': [],
                        'fact_check_scores': []
                    }
                
                source_data[source]['articles'].append(article)
                source_data[source]['credibility_scores'].append(article.credibility_score)
                source_data[source]['bias_scores'].append(article.bias_score)
                source_data[source]['fact_check_scores'].append(article.fact_check_score)
            
            # Calculate source statistics
            source_comparison = {}
            for source, data in source_data.items():
                source_comparison[source] = {
                    'article_count': len(data['articles']),
                    'avg_credibility': np.mean(data['credibility_scores']),
                    'avg_bias': np.mean(data['bias_scores']),
                    'avg_fact_check': np.mean(data['fact_check_scores']),
                    'credibility_std': np.std(data['credibility_scores']),
                    'bias_std': np.std(data['bias_scores']),
                    'consistency_score': 1.0 - np.std(data['fact_check_scores'])
                }
            
            return source_comparison
            
        except Exception as e:
            st.error(f"Error comparing sources: {str(e)}")
            return {}
    
    def get_system_health(self) -> Dict:
        """Get system health and performance metrics"""
        try:
            # Database statistics
            stats = self.database_manager.get_statistics()
            
            # Vector database info
            collection_info = self.database_manager.get_collection_info()
            
            # Model information
            embedding_info = self.embedding_engine.get_model_info_display()
            
            # Recent processing performance
            recent_articles = self.database_manager.get_recent_articles(limit=10)
            
            health_status = {
                'database': {
                    'total_articles': stats.total_articles,
                    'vector_count': collection_info.get('count', 0),
                    'sources_active': len(stats.source_distribution),
                    'last_update': stats.latest_update
                },
                'models': {
                    'embedding_model': embedding_info,
                    'bias_detector': 'Active' if self.bias_detector.bias_classifier else 'Limited',
                    'fact_checker': 'Active'
                },
                'performance': {
                    'avg_credibility': stats.avg_credibility,
                    'avg_bias_score': stats.avg_bias_score,
                    'avg_fact_check_score': stats.avg_fact_check_score,
                    'recent_articles': len(recent_articles)
                },
                'system_status': 'Healthy' if stats.total_articles > 0 else 'Needs Data',
                'timestamp': datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                'error': str(e),
                'system_status': 'Error',
                'timestamp': datetime.now().isoformat()
            }
    
    def cleanup_old_data(self, days: int = 30) -> Dict:
        """Clean up old articles and data"""
        try:
            deleted_count = self.database_manager.cleanup_old_articles(days)
            
            return {
                'deleted_articles': deleted_count,
                'cleanup_date': datetime.now().isoformat(),
                'retention_days': days
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_data(self, format: str = 'json', filters: Dict = None) -> Optional[str]:
        """Export system data in various formats"""
        try:
            # Get articles based on filters
            if filters:
                # Apply filters to get specific articles
                articles = self._get_filtered_articles(filters)
            else:
                articles = self.database_manager.get_recent_articles(limit=1000)
            
            if format.lower() == 'json':
                import json
                data = {
                    'articles': [article.to_dict() for article in articles],
                    'export_timestamp': datetime.now().isoformat(),
                    'total_articles': len(articles)
                }
                return json.dumps(data, indent=2)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                article_data = []
                for article in articles:
                    article_data.append({
                        'id': article.id,
                        'title': article.title,
                        'source': article.source,
                        'published_at': article.published_at.isoformat(),
                        'credibility_score': article.credibility_score,
                        'bias_score': article.bias_score,
                        'fact_check_score': article.fact_check_score,
                        'url': article.url
                    })
                
                df = pd.DataFrame(article_data)
                return df.to_csv(index=False)
            
            else:
                return None
                
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None
    
    def _get_filtered_articles(self, filters: Dict) -> List[NewsArticle]:
        """Get articles based on filters"""
        # This would implement more sophisticated filtering
        # For now, return recent articles
        return self.database_manager.get_recent_articles(limit=100)
    
    def get_article_recommendations(self, article_id: str, n_recommendations: int = 5) -> List[SearchResult]:
        """Get recommended articles based on similarity"""
        try:
            # Get the source article
            source_article = self.database_manager.get_article_by_id(article_id)
            if not source_article:
                return []
            
            # Get its embedding from vector database
            vector_results = self.database_manager.collection.get(ids=[article_id])
            
            if not vector_results['documents'] or len(vector_results['documents']) == 0:
                return []
            
            # Use the content as a query to find similar articles
            content = vector_results['documents'][0]
            
            # Generate embedding for the content
            content_embedding = self.embedding_engine.encode_text(content)
            
            if content_embedding is None:
                return []
            
            # Search for similar articles (excluding the source article)
            similar_results = self.database_manager.search_articles(
                content_embedding.tolist(), 
                n_recommendations + 1
            )
            
            # Filter out the source article
            recommendations = [r for r in similar_results if r.article_id != article_id]
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def batch_fact_check(self, article_ids: List[str]) -> Dict[str, List[Dict]]:
        """Perform batch fact-checking on multiple articles"""
        results = {}
        
        for article_id in article_ids:
            try:
                article = self.database_manager.get_article_by_id(article_id)
                if article:
                    fact_check_results = self.fact_checker.fact_check_article(article)
                    results[article_id] = [r.to_dict() for r in fact_check_results]
                else:
                    results[article_id] = []
                    
            except Exception as e:
                results[article_id] = [{'error': str(e)}]
        
        return results
    
    def get_statistics(self) -> SystemStatistics:
        """Get comprehensive system statistics"""
        return self.database_manager.get_statistics()
    
    def is_healthy(self) -> bool:
        """Check if the system is healthy and operational"""
        try:
            # Check if all components are initialized
            if not self._initialized:
                return False
            
            # Check if we have data
            stats = self.get_statistics()
            if stats.total_articles == 0:
                return False
            
            # Check if models are loaded
            if self.embedding_engine.model is None:
                return False
            
            return True
            
        except Exception:
            return False