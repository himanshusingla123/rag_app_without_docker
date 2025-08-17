"""
UI components for the News RAG System
"""
import streamlit as st
# import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from models.data_models import SearchResult, NewsArticle


class UIComponents:
    """Reusable UI components for the News RAG System"""
    
    def __init__(self):
        self.colors = {
            'high': '#4CAF50',      # Green
            'medium': '#FF9800',    # Orange  
            'low': '#F44336',       # Red
            'neutral': '#9E9E9E'    # Grey
        }
    
    def render_search_results(self, search_results: List[SearchResult], rag_system):
        """Render search results with comprehensive analysis"""
        st.subheader(f"🎯 Search Results ({len(search_results)} found)")
        
        for result in search_results:
            metadata = result.metadata
            
            with st.expander(
                f"Rank #{result.rank} - {metadata.get('source', 'Unknown')} "
                f"(Relevance: {result.relevance_score:.2f})",
                expanded=result.rank <= 3  # Expand top 3 results
            ):
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    credibility = metadata.get('credibility_score', 0.0)
                    self._render_score_metric("Credibility", credibility)
                
                with col2:
                    bias = metadata.get('bias_score', 0.0)
                    self._render_score_metric("Bias Score", bias, reverse=True)
                
                with col3:
                    fact_check = metadata.get('fact_check_score', 0.0)
                    self._render_score_metric("Fact-Check", fact_check)
                
                with col4:
                    self._render_score_metric("Relevance", result.relevance_score)
                
                # Content preview
                st.write("**Content Preview:**")
                content_preview = self._truncate_content(result.content, 500)
                st.write(content_preview)
                
                # Action buttons
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if metadata.get('url'):
                        st.markdown(f"[📖 Read Full Article]({metadata['url']})")
                
                with col2:
                    if st.button("🔍Detailed Analysis", key=f"analyze_{result.rank}"):
                        if result.article_id:
                            self._show_detailed_analysis(result.article_id, rag_system)
                
                with col3:
                    if st.button("💡 Recommendations", key=f"recommend_{result.rank}"):
                        if result.article_id:
                            self._show_recommendations(result.article_id, rag_system)
                
                # Additional metadata
                if metadata.get('published_at'):
                    try:
                        pub_date = datetime.fromisoformat(metadata['published_at'].replace('Z', '+00:00'))
                        st.caption(f"Published: {pub_date.strftime('%Y-%m-%d %H:%M')}")
                    except Exception:
                        st.caption(f"Published: {metadata['published_at']}")
    
    def _render_score_metric(self, label: str, score: float, reverse: bool = False):
        """Render a score metric with color coding"""
        # Determine color based on score
        if reverse:  # For bias scores, lower is better
            if score < 0.3:
                color = self.colors['high']
            elif score < 0.6:
                color = self.colors['medium']
            else:
                color = self.colors['low']
        else:  # For credibility/fact-check, higher is better
            if score > 0.7:
                color = self.colors['high']
            elif score > 0.4:
                color = self.colors['medium']
            else:
                color = self.colors['low']
        
        st.markdown(
            f"<div style='text-align: center; padding: 10px; border-radius: 5px; "
            f"background-color: {color}20; border: 1px solid {color};'>"
            f"<strong>{label}</strong><br>"
            f"<span style='color: {color}; font-size: 1.2em;'>{score:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    def _truncate_content(self, content: str, max_length: int = 500) -> str:
        """Truncate content for preview"""
        if len(content) <= max_length:
            return content
        
        truncated = content[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def _show_detailed_analysis(self, article_id: str, rag_system):
        """Show detailed analysis for an article"""
        try:
            analysis = rag_system.get_article_analysis(article_id)
            
            if 'error' in analysis:
                st.error(f"Analysis error: {analysis['error']}")
                return
            
            st.subheader("🔬 Detailed Analysis")
            
            # Bias analysis
            if 'bias_analysis' in analysis:
                bias = analysis['bias_analysis']
                st.write("**Bias Analysis:**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Political", f"{bias['political_bias']:.2f}")
                with col2:
                    st.metric("Emotional", f"{bias['emotional_bias']:.2f}")
                with col3:
                    st.metric("Toxic", f"{bias['toxic_bias']:.2f}")
                with col4:
                    st.metric("Complexity", f"{bias['readability_complexity']:.2f}")
            
            # Fact-check results
            if 'fact_check_results' in analysis and analysis['fact_check_results']:
                st.write("**Fact-Check Results:**")
                for i, result in enumerate(analysis['fact_check_results'][:3]):
                    with st.expander(f"Claim {i+1}: {result['verdict']}"):
                        st.write(f"**Claim:** {result['claim']}")
                        st.write(f"**Verdict:** {result['verdict']}")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                        st.write(f"**Reasoning:** {result['reasoning']}")
            
            # Keywords and statistics
            col1, col2 = st.columns(2)
            with col1:
                if 'keywords' in analysis:
                    st.write("**Key Topics:**")
                    st.write(", ".join(analysis['keywords'][:10]))
            
            with col2:
                if 'statistics' in analysis:
                    st.write("**Statistics Found:**")
                    st.write(", ".join(analysis['statistics'][:5]))
                    
        except Exception as e:
            st.error(f"Error showing detailed analysis: {str(e)}")
    
    def _show_recommendations(self, article_id: str, rag_system):
        """Show article recommendations"""
        try:
            recommendations = rag_system.get_article_recommendations(article_id, 5)
            
            if not recommendations:
                st.info("No recommendations available")
                return
            
            st.subheader("💡 Recommended Articles")
            
            for i, rec in enumerate(recommendations):
                metadata = rec.metadata
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{metadata.get('source', 'Unknown')}**")
                    preview = self._truncate_content(rec.content, 100)
                    st.write(preview)
                
                with col2:
                    st.metric("Similarity", f"{rec.relevance_score:.2f}")
                
                with col3:
                    if metadata.get('url'):
                        st.markdown(f"[📖 Read]({metadata['url']})")
                        
        except Exception as e:
            st.error(f"Error showing recommendations: {str(e)}")
    
    def render_recent_articles(self, articles: List[NewsArticle], rag_system):
        """Render recent articles feed"""
        for article in articles:
            with st.container():
                # Article header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{article.source}**: {article.title}")
                
                with col2:
                    st.write(f"📅 {article.published_at.strftime('%m/%d %H:%M')}")
                
                with col3:
                    if st.button("📖 Details", key=f"details_{article.id}"):
                        self._show_article_details(article, rag_system)
                
                # Quick metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Credibility: {article.credibility_score:.2f}")
                with col2:
                    st.caption(f"Bias: {article.bias_score:.2f}")
                with col3:
                    st.caption(f"Fact-Check: {article.fact_check_score:.2f}")
                
                st.markdown("---")
    
    def _show_article_details(self, article: NewsArticle, rag_system):
        """Show detailed view of an article"""
        st.subheader(f"📰 {article.title}")
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Source:** {article.source}")
        with col2:
            st.write(f"**Published:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")
        with col3:
            if article.author:
                st.write(f"**Author:** {article.author}")
        
        # Scores
        col1, col2, col3 = st.columns(3)
        with col1:
            self._render_score_metric("Credibility", article.credibility_score)
        with col2:
            self._render_score_metric("Bias Score", article.bias_score, reverse=True)
        with col3:
            self._render_score_metric("Fact-Check", article.fact_check_score)
        
        # Content
        if article.description:
            st.write("**Summary:**")
            st.write(article.description)
        
        if article.url:
            st.markdown(f"[📖 Read Full Article]({article.url})")
    
    def render_system_health(self, health: Dict[str, Any]):
        """Render system health dashboard"""
        status = health.get('system_status', 'Unknown')
        
        if status == 'Healthy':
            st.success(f"🟢 System Status: {status}")
        elif status == 'Needs Data':
            st.warning(f"🟡 System Status: {status}")
        else:
            st.error(f"🔴 System Status: {status}")
        
        # Database health
        if 'database' in health:
            db = health['database']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", db.get('total_articles', 0))
            with col2:
                st.metric("Vector Count", db.get('vector_count', 0))
            with col3:
                st.metric("Active Sources", db.get('sources_active', 0))
            with col4:
                last_update = db.get('last_update', 'Never')
                if last_update and last_update != 'Never':
                    try:
                        update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        st.metric("Last Update", update_time.strftime('%m/%d %H:%M'))
                    except Exception:
                        st.metric("Last Update", "Unknown")
                else:
                    st.metric("Last Update", "Never")
        
        # Model status
        if 'models' in health:
            models = health['models']
            st.subheader("🤖 Model Status")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                embedding_status = models.get('embedding_model', {}).get('status', 'Unknown')
                if embedding_status == 'Loaded successfully':
                    st.success("✅ Embeddings")
                else:
                    st.error("❌ Embeddings")
            
            with col2:
                bias_status = models.get('bias_detector', 'Unknown')
                if bias_status == 'Active':
                    st.success("✅ Bias Detection")
                else:
                    st.warning("⚠️ Bias Detection")
            
            with col3:
                fact_check_status = models.get('fact_checker', 'Unknown')
                if fact_check_status == 'Active':
                    st.success("✅ Fact Checking")
                else:
                    st.error("❌ Fact Checking")
        
        # Performance metrics
        if 'performance' in health:
            perf = health['performance']
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Avg Credibility", 
                    f"{perf.get('avg_credibility', 0):.2f}",
                    help="Average credibility across all articles"
                )
            with col2:
                st.metric(
                    "Avg Bias Score", 
                    f"{perf.get('avg_bias_score', 0):.2f}",
                    help="Average bias score (lower is better)"
                )
            with col3:
                st.metric(
                    "Avg Fact-Check", 
                    f"{perf.get('avg_fact_check_score', 0):.2f}",
                    help="Average fact-checking score"
                )
    
    def render_error_message(self, error: str, context: str = ""):
        """Render standardized error message"""
        error_container = st.container()
        
        with error_container:
            st.error("❌ An error occurred")
            
            with st.expander("Error Details"):
                if context:
                    st.write(f"**Context:** {context}")
                st.write(f"**Error:** {error}")
                st.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def render_loading_spinner(self, message: str = "Processing..."):
        """Render loading spinner with message"""
        return st.spinner(message)
    
    def render_success_message(self, message: str, details: str = ""):
        """Render success message"""
        st.success(f"✅ {message}")
        if details:
            st.info(details)
    
    def render_warning_message(self, message: str, details: str = ""):
        """Render warning message"""
        st.warning(f"⚠️ {message}")
        if details:
            st.caption(details)
    
    def render_info_panel(self, title: str, content: Dict[str, Any]):
        """Render informational panel"""
        with st.expander(f"ℹ️ {title}"):
            for key, value in content.items():
                if isinstance(value, dict):
                    st.json(value)
                elif isinstance(value, list):
                    st.write(f"**{key}:**")
                    for item in value[:10]:  # Limit to 10 items
                        st.write(f"- {item}")
                else:
                    st.write(f"**{key}:** {value}")
    
    def render_metric_card(self, title: str, value: str, delta: str = None, 
                          help_text: str = None):
        """Render a metric card"""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )