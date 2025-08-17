"""
Main Streamlit application for the News RAG System
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

# Import our modular components
from core.rag_system import NewsRAGSystem
from ui.components import UIComponents
from ui.dashboard import Dashboard
from config import PAGE_TITLE, PAGE_ICON, DEFAULT_SEARCH_RESULTS, MAX_SEARCH_RESULTS


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'search'


def load_system():
    """Load and initialize the RAG system"""
    if st.session_state.rag_system is None:
        with st.spinner("Initializing News RAG System..."):
            try:
                rag_system = NewsRAGSystem()
                if rag_system.initialize_system():
                    st.session_state.rag_system = rag_system
                    st.success("System initialized successfully!")
                else:
                    st.error("Failed to initialize system. Please check the logs.")
                    return False
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
                return False
    
    return True


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🔍 News RAG System with Fact-Checking")
    st.markdown("*Intelligent news analysis with credibility assessment and bias detection*")
    
    # Load system
    if not load_system():
        st.stop()
    
    rag_system = st.session_state.rag_system
    ui_components = UIComponents()
    dashboard = Dashboard()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    view_options = {
        "🔍 Search & Analysis": "search",
        "📊 Dashboard": "dashboard", 
        "📡 Real-time Monitoring": "monitoring",
        "⚙️ System Management": "management",
        "📈 Analytics": "analytics"
    }
    
    selected_view = st.sidebar.selectbox(
        "Select View",
        options=list(view_options.keys()),
        index=0
    )
    
    current_view = view_options[selected_view]
    st.session_state.current_view = current_view
    
    # System health indicator
    with st.sidebar:
        st.markdown("---")
        st.subheader("System Status")
        
        if rag_system.is_healthy():
            st.success("🟢 System Healthy")
        else:
            st.warning("🟡 System Issues")
        
        # Quick stats
        try:
            stats = rag_system.get_statistics()
            st.metric("Total Articles", stats.total_articles)
            st.metric("Avg Credibility", f"{stats.avg_credibility:.2f}")
        except Exception:
            st.info("Loading statistics...")
    
    # Main content area based on selected view
    if current_view == "search":
        render_search_view(rag_system, ui_components)
    
    elif current_view == "dashboard":
        render_dashboard_view(rag_system, dashboard)
    
    elif current_view == "monitoring":
        render_monitoring_view(rag_system, ui_components)
    
    elif current_view == "management":
        render_management_view(rag_system, ui_components)
    
    elif current_view == "analytics":
        render_analytics_view(rag_system, dashboard)


def render_search_view(rag_system, ui_components):
    """Render the search and analysis view"""
    st.header("🔍 Search & Fact-Check News")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., climate change, election results, vaccine efficacy",
            help="Search for news articles and get comprehensive analysis"
        )
    
    with col2:
        n_results = st.slider(
            "Results", 
            1, 
            MAX_SEARCH_RESULTS, 
            DEFAULT_SEARCH_RESULTS,
            help="Number of search results to return"
        )
    
    # Search filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_credibility = st.slider("Min Credibility", 0.0, 1.0, 0.0, 0.1)
        with col2:
            max_bias = st.slider("Max Bias Score", 0.0, 1.0, 1.0, 0.1)
        with col3:
            min_fact_check = st.slider("Min Fact-Check Score", 0.0, 1.0, 0.0, 0.1)
        
        # Source filter
        try:
            stats = rag_system.get_statistics()
            available_sources = list(stats.source_distribution.keys())
            selected_sources = st.multiselect(
                "Filter by Sources",
                available_sources,
                default=[],
                help="Select specific news sources"
            )
        except Exception:
            selected_sources = []
    
    # Search button and results
    if st.button("🔍 Search", type="primary", disabled=not search_query):
        if search_query:
            # Add to search history
            if search_query not in st.session_state.search_history:
                st.session_state.search_history.append(search_query)
                if len(st.session_state.search_history) > 10:
                    st.session_state.search_history.pop(0)
            
            with st.spinner("Searching articles..."):
                # Prepare filters
                filters = {
                    'min_credibility': min_credibility,
                    'max_bias': max_bias,
                    'min_fact_check': min_fact_check,
                    'sources': selected_sources if selected_sources else None
                }
                
                search_results = rag_system.search_articles(
                    search_query, 
                    n_results, 
                    filters
                )
                
                if search_results:
                    ui_components.render_search_results(search_results, rag_system)
                else:
                    st.warning("No relevant articles found. Try a different search query or adjust filters.")
    
    # Search history
    if st.session_state.search_history:
        st.subheader("Recent Searches")
        for query in reversed(st.session_state.search_history[-5:]):
            if st.button(f"📄 {query}", key=f"history_{query}"):
                st.experimental_rerun()


def render_dashboard_view(rag_system, dashboard):
    """Render the dashboard view"""
    st.header("📊 System Dashboard")
    
    try:
        # System statistics
        stats = rag_system.get_statistics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Articles", 
                stats.total_articles,
                help="Total number of processed articles"
            )
        with col2:
            st.metric(
                "Avg Credibility", 
                f"{stats.avg_credibility:.2f}",
                help="Average credibility score across all articles"
            )
        with col3:
            st.metric(
                "Avg Bias Score", 
                f"{stats.avg_bias_score:.2f}",
                help="Average bias score (lower is better)"
            )
        with col4:
            st.metric(
                "Avg Fact-Check", 
                f"{stats.avg_fact_check_score:.2f}",
                help="Average fact-checking score"
            )
        
        # Charts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Source distribution
            st.subheader("📰 Source Distribution")
            if stats.source_distribution:
                source_df = pd.DataFrame(
                    list(stats.source_distribution.items()),
                    columns=['Source', 'Articles']
                )
                st.bar_chart(source_df.set_index('Source'))
        
        with col2:
            # Trending topics
            st.subheader("🔥 Trending Topics")
            trending = rag_system.get_trending_topics(days=7, top_k=10)
            if trending:
                trending_df = pd.DataFrame(trending)
                st.bar_chart(trending_df.set_index('topic')['frequency'])
            else:
                st.info("No trending topics available")
        
        # Source comparison
        st.subheader("🆚 Source Comparison")
        source_comparison = rag_system.compare_sources(days=7)
        if source_comparison:
            dashboard.render_source_comparison(source_comparison)
        else:
            st.info("Not enough data for source comparison")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")


def render_monitoring_view(rag_system, ui_components):
    """Render the real-time monitoring view"""
    st.header("📡 Real-time Monitoring")
    
    # News ingestion controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("News Ingestion")
        max_articles = st.number_input(
            "Max articles to fetch", 
            min_value=10, 
            max_value=200, 
            value=50,
            help="Maximum number of articles to fetch from all sources"
        )
    
    with col2:
        st.subheader("Actions")
        if st.button("🔄 Fetch Latest News", type="primary"):
            with st.spinner("Fetching and processing news..."):
                job = rag_system.process_news_batch(max_articles)
                
                if job.status == 'completed':
                    st.success(f"Successfully processed {job.processed_articles} articles!")
                    st.session_state.last_update = datetime.now()
                    
                    # Show processing summary
                    if job.errors:
                        with st.expander(f"⚠️ Processing Errors ({len(job.errors)})"):
                            for error in job.errors:
                                st.warning(error)
                else:
                    st.error("Failed to process articles")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable Auto-refresh (30 minutes)")
    
    if auto_refresh:
        # Check if it's time for auto-refresh
        if (st.session_state.last_update is None or 
            datetime.now() - st.session_state.last_update > timedelta(minutes=30)):
            
            st.info("Auto-refreshing news feed...")
            time.sleep(2)  # Small delay for UI
            st.experimental_rerun()
    
    # System health monitoring
    st.subheader("🏥 System Health")
    health = rag_system.get_system_health()
    ui_components.render_system_health(health)
    
    # Real-time feed
    st.subheader("📊 Latest Articles")
    try:
        recent_articles = rag_system.database_manager.get_recent_articles(limit=10)
        if recent_articles:
            ui_components.render_recent_articles(recent_articles, rag_system)
        else:
            st.info("No recent articles available. Click 'Fetch Latest News' to get started.")
    except Exception as e:
        st.error(f"Error loading recent articles: {str(e)}")


def render_management_view(rag_system, ui_components):
    """Render the system management view"""
    st.header("⚙️ System Management")
    
    # Data management
    st.subheader("🗄️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Cleanup Old Data**")
        cleanup_days = st.number_input(
            "Remove articles older than (days)", 
            min_value=1, 
            max_value=365, 
            value=30
        )
        
        if st.button("🧹 Cleanup Old Articles"):
            with st.spinner("Cleaning up old articles..."):
                result = rag_system.cleanup_old_data(cleanup_days)
                if 'error' not in result:
                    st.success(f"Removed {result['deleted_articles']} old articles")
                else:
                    st.error(f"Cleanup failed: {result['error']}")
    
    with col2:
        st.write("**Export Data**")
        export_format = st.selectbox("Format", ["JSON", "CSV"])
        
        if st.button("📤 Export Data"):
            with st.spinner("Exporting data..."):
                exported_data = rag_system.export_data(export_format.lower())
                if exported_data:
                    st.download_button(
                        label=f"Download {export_format}",
                        data=exported_data,
                        file_name=f"news_data_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                        mime="application/json" if export_format == "JSON" else "text/csv"
                    )
                else:
                    st.error("Export failed")
    
    with col3:
        st.write("**System Info**")
        if st.button("ℹ️ Show System Info"):
            health = rag_system.get_system_health()
            st.json(health)
    
    # Configuration
    st.subheader("🔧 Configuration")
    
    with st.expander("RSS Feed Sources"):
        sources = rag_system.ingestion_engine.get_available_sources()
        for source in sources:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{source['name']}**")
            with col2:
                st.write(f"Credibility: {source['credibility']:.2f}")
            with col3:
                st.write(f"Status: {source['status']}")
    
    # Model information
    st.subheader("🤖 Model Information")
    with st.expander("Embedding Model"):
        model_info = rag_system.embedding_engine.get_model_info_display()
        st.json(model_info)


def render_analytics_view(rag_system, dashboard):
    """Render the analytics view"""
    st.header("📈 Advanced Analytics")
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.selectbox(
            "Analysis Period",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Source Comparison", "Bias Analysis", "Fact-Check Trends", "Topic Analysis"]
        )
    
    try:
        if analysis_type == "Source Comparison":
            st.subheader("📊 Source Credibility Comparison")
            source_comparison = rag_system.compare_sources(days=7)
            if source_comparison:
                dashboard.render_detailed_source_analysis(source_comparison)
            else:
                st.info("Not enough data for source comparison")
        
        elif analysis_type == "Bias Analysis":
            st.subheader("🎭 Bias Distribution Analysis")
            # Implementation for bias analysis
            st.info("Bias analysis visualization coming soon")
        
        elif analysis_type == "Fact-Check Trends":
            st.subheader("✅ Fact-Checking Trends")
            # Implementation for fact-check trends
            st.info("Fact-check trends visualization coming soon")
        
        elif analysis_type == "Topic Analysis":
            st.subheader("🏷️ Topic Clustering")
            trending = rag_system.get_trending_topics(days=7, top_k=20)
            if trending:
                dashboard.render_topic_analysis(trending)
            else:
                st.info("No topic data available")
                
    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🛠️ **Built with:** Streamlit, ChromaDB, Sentence Transformers, RSS Feeds"
    )


if __name__ == "__main__":
    main()