"""
Dashboard components for advanced visualizations and analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any


class Dashboard:
    """Advanced dashboard components for the News RAG System"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def render_source_comparison(self, source_data: Dict[str, Dict[str, float]]):
        """Render source comparison visualization"""
        if not source_data:
            st.info("No source data available")
            return
        
        # Prepare data for visualization
        sources = list(source_data.keys())
        metrics = ['avg_credibility', 'avg_bias', 'avg_fact_check']
        
        df_data = []
        for source, data in source_data.items():
            df_data.append({
                'Source': source,
                'Credibility': data.get('avg_credibility', 0),
                'Bias Score': data.get('avg_bias', 0),
                'Fact-Check Score': data.get('avg_fact_check', 0),
                'Article Count': data.get('article_count', 0),
                'Consistency': data.get('consistency_score', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Detailed Metrics", "🎯 Radar Chart"])
        
        with tab1:
            self._render_source_overview(df)
        
        with tab2:
            self._render_detailed_metrics(df)
        
        with tab3:
            self._render_source_radar_chart(df)
    
    def _render_source_overview(self, df: pd.DataFrame):
        """Render source overview charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Credibility comparison
            fig_cred = px.bar(
                df.sort_values('Credibility', ascending=False),
                x='Source',
                y='Credibility',
                title='Source Credibility Comparison',
                color='Credibility',
                color_continuous_scale='RdYlGn'
            )
            fig_cred.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cred, use_container_width=True)
        
        with col2:
            # Bias comparison (lower is better)
            fig_bias = px.bar(
                df.sort_values('Bias Score', ascending=True),
                x='Source',
                y='Bias Score',
                title='Source Bias Comparison (Lower is Better)',
                color='Bias Score',
                color_continuous_scale='RdYlGn_r'
            )
            fig_bias.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bias, use_container_width=True)
    
    def _render_detailed_metrics(self, df: pd.DataFrame):
        """Render detailed metrics comparison"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Credibility vs Bias', 'Fact-Check Distribution', 
                           'Article Count', 'Consistency Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter plot: Credibility vs Bias
        fig.add_trace(
            go.Scatter(
                x=df['Bias Score'],
                y=df['Credibility'],
                mode='markers+text',
                text=df['Source'],
                textposition="top center",
                marker=dict(
                    size=df['Article Count']/2,
                    color=df['Fact-Check Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Fact-Check Score")
                ),
                name='Sources'
            ),
            row=1, col=1
        )
        
        # Fact-check distribution
        fig.add_trace(
            go.Bar(
                x=df['Source'],
                y=df['Fact-Check Score'],
                name='Fact-Check Score',
                marker_color=self.color_palette['info']
            ),
            row=1, col=2
        )
        
        # Article count
        fig.add_trace(
            go.Bar(
                x=df['Source'],
                y=df['Article Count'],
                name='Article Count',
                marker_color=self.color_palette['secondary']
            ),
            row=2, col=1
        )
        
        # Consistency scores
        fig.add_trace(
            go.Bar(
                x=df['Source'],
                y=df['Consistency'],
                name='Consistency',
                marker_color=self.color_palette['success']
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_source_radar_chart(self, df: pd.DataFrame):
        """Render radar chart for source comparison"""
        fig = go.Figure()
        
        # Normalize bias scores (invert so higher is better)
        df_norm = df.copy()
        df_norm['Bias Score'] = 1 - df_norm['Bias Score']  # Invert bias
        df_norm['Consistency'] = df_norm['Consistency'].fillna(0)
        
        categories = ['Credibility', 'Bias Score', 'Fact-Check Score', 'Consistency']
        
        for idx, row in df_norm.iterrows():
            values = [row['Credibility'], row['Bias Score'], 
                     row['Fact-Check Score'], row['Consistency']]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=row['Source'],
                line=dict(width=2),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Source Quality Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_source_analysis(self, source_data: Dict[str, Dict[str, float]]):
        """Render detailed source analysis"""
        self.render_source_comparison(source_data)
        
        # Additional analysis
        st.subheader("📊 Source Rankings")
        
        # Create ranking dataframe
        ranking_data = []
        for source, data in source_data.items():
            # Calculate composite score
            credibility = data.get('avg_credibility', 0)
            bias = 1 - data.get('avg_bias', 0)  # Invert bias (lower is better)
            fact_check = data.get('avg_fact_check', 0)
            consistency = data.get('consistency_score', 0)
            
            composite_score = (credibility * 0.3 + bias * 0.3 + 
                             fact_check * 0.3 + consistency * 0.1)
            
            ranking_data.append({
                'Source': source,
                'Composite Score': composite_score,
                'Credibility': credibility,
                'Low Bias': bias,
                'Fact-Check': fact_check,
                'Consistency': consistency,
                'Articles': data.get('article_count', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Composite Score', ascending=False)
        
        # Display ranking table
        st.dataframe(
            ranking_df.style.format({
                'Composite Score': '{:.3f}',
                'Credibility': '{:.3f}',
                'Low Bias': '{:.3f}',
                'Fact-Check': '{:.3f}',
                'Consistency': '{:.3f}'
            }).background_gradient(subset=['Composite Score']),
            use_container_width=True
        )
    
    def render_topic_analysis(self, trending_topics: List[Dict[str, Any]]):
        """Render topic analysis visualization"""
        if not trending_topics:
            st.info("No topic data available")
            return
        
        df = pd.DataFrame(trending_topics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top topics bar chart
            fig_bar = px.bar(
                df.head(15),
                x='frequency',
                y='topic',
                orientation='h',
                title='Top 15 Trending Topics',
                color='trend_score',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Topic frequency distribution
            fig_hist = px.histogram(
                df,
                x='frequency',
                nbins=20,
                title='Topic Frequency Distribution',
                color_discrete_sequence=[self.color_palette['primary']]
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Topic cloud visualization (simplified)
        st.subheader("🏷️ Topic Cloud")
        
        # Create a simple topic cloud using text sizing
        topic_text = ""
        for topic in df.head(20).itertuples():
            size = min(6, max(1, topic.frequency // 2))
            topic_text += f"<span style='font-size: {size}em; margin: 10px; color: {self.color_palette['info']};'>{topic.topic}</span> "
        
        st.markdown(
            f"<div style='text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 10px;'>"
            f"{topic_text}"
            f"</div>",
            unsafe_allow_html=True
        )
    
    def render_time_series_analysis(self, time_series_data: List[Dict[str, Any]]):
        """Render time series analysis of articles"""
        if not time_series_data:
            st.info("No time series data available")
            return
        
        df = pd.DataFrame(time_series_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create multiple time series charts
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Articles Published Over Time', 
                           'Average Credibility Over Time',
                           'Average Bias Score Over Time'),
            vertical_spacing=0.1
        )
        
        # Article count over time
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['article_count'],
                mode='lines+markers',
                name='Article Count',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Credibility over time
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['avg_credibility'],
                mode='lines+markers',
                name='Avg Credibility',
                line=dict(color=self.color_palette['success'])
            ),
            row=2, col=1
        )
        
        # Bias over time
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['avg_bias'],
                mode='lines+markers',
                name='Avg Bias',
                line=dict(color=self.color_palette['warning'])
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=900, showlegend=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_bias_distribution(self, bias_data: List[Dict[str, float]]):
        """Render bias distribution analysis"""
        if not bias_data:
            st.info("No bias data available")
            return
        
        df = pd.DataFrame(bias_data)
        
        # Create bias distribution charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Political Bias Distribution', 'Emotional Bias Distribution',
                           'Toxic Content Distribution', 'Overall Bias Distribution')
        )
        
        # Political bias
        fig.add_trace(
            go.Histogram(
                x=df['political_bias'],
                name='Political Bias',
                nbinsx=20,
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # Emotional bias
        fig.add_trace(
            go.Histogram(
                x=df['emotional_bias'],
                name='Emotional Bias',
                nbinsx=20,
                marker_color=self.color_palette['secondary']
            ),
            row=1, col=2
        )
        
        # Toxic content
        fig.add_trace(
            go.Histogram(
                x=df['toxic_bias'],
                name='Toxic Content',
                nbinsx=20,
                marker_color=self.color_palette['warning']
            ),
            row=2, col=1
        )
        
        # Overall bias
        fig.add_trace(
            go.Histogram(
                x=df['overall_bias_score'],
                name='Overall Bias',
                nbinsx=20,
                marker_color=self.color_palette['info']
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Bias Score", row=2, col=1)
        fig.update_xaxes(title_text="Bias Score", row=2, col=2)
        fig.update_yaxes(title_text="Frequency")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Bias Statistics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Political Bias", 
                f"{df['political_bias'].mean():.3f}",
                delta=f"{df['political_bias'].std():.3f} std"
            )
        
        with col2:
            st.metric(
                "Avg Emotional Bias", 
                f"{df['emotional_bias'].mean():.3f}",
                delta=f"{df['emotional_bias'].std():.3f} std"
            )
        
        with col3:
            st.metric(
                "Avg Toxic Content", 
                f"{df['toxic_bias'].mean():.3f}",
                delta=f"{df['toxic_bias'].std():.3f} std"
            )
        
        with col4:
            st.metric(
                "Avg Overall Bias", 
                f"{df['overall_bias_score'].mean():.3f}",
                delta=f"{df['overall_bias_score'].std():.3f} std"
            )
    
    def render_fact_check_trends(self, fact_check_data: List[Dict[str, Any]]):
        """Render fact-checking trends visualization"""
        if not fact_check_data:
            st.info("No fact-check data available")
            return
        
        df = pd.DataFrame(fact_check_data)
        
        # Verdict distribution
        col1, col2 = st.columns(2)
        
        with col1:
            verdict_counts = df['verdict'].value_counts()
            fig_pie = px.pie(
                values=verdict_counts.values,
                names=verdict_counts.index,
                title='Fact-Check Verdict Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution by verdict
            fig_box = px.box(
                df,
                x='verdict',
                y='confidence',
                title='Confidence Distribution by Verdict'
            )
            fig_box.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Fact-check trends over time if date available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            daily_verdicts = df.groupby(['date', 'verdict']).size().unstack(fill_value=0)
            
            fig_area = px.area(
                daily_verdicts.reset_index(),
                x='date',
                y=daily_verdicts.columns,
                title='Fact-Check Trends Over Time'
            )
            st.plotly_chart(fig_area, use_container_width=True)
    
    def render_performance_dashboard(self, performance_data: Dict[str, Any]):
        """Render system performance dashboard"""
        if not performance_data:
            st.info("No performance data available")
            return
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            processing_rate = performance_data.get('processing_rate', 0)
            st.metric(
                "Processing Rate",
                f"{processing_rate:.1f} articles/min",
                help="Average articles processed per minute"
            )
        
        with col2:
            accuracy = performance_data.get('accuracy', 0)
            st.metric(
                "System Accuracy",
                f"{accuracy:.1%}",
                help="Overall system accuracy"
            )
        
        with col3:
            uptime = performance_data.get('uptime', 0)
            st.metric(
                "System Uptime",
                f"{uptime:.1%}",
                help="System availability percentage"
            )
        
        with col4:
            error_rate = performance_data.get('error_rate', 0)
            st.metric(
                "Error Rate",
                f"{error_rate:.2%}",
                help="Percentage of processing errors"
            )
        
        # Performance trends
        if 'trends' in performance_data:
            trends = performance_data['trends']
            
            fig = go.Figure()
            
            if 'timestamps' in trends:
                timestamps = pd.to_datetime(trends['timestamps'])
                
                # Add multiple metrics
                metrics = ['processing_rate', 'accuracy', 'memory_usage']
                colors = [self.color_palette['primary'], 
                         self.color_palette['success'], 
                         self.color_palette['warning']]
                
                for metric, color in zip(metrics, colors):
                    if metric in trends:
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=trends[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title(),
                            line=dict(color=color)
                        ))
                
                fig.update_layout(
                    title='System Performance Trends',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_matrix(self, correlation_data: pd.DataFrame):
        """Render correlation matrix heatmap"""
        if correlation_data.empty:
            st.info("No correlation data available")
            return
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Article Metrics Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            width=600,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("🔍 Key Correlations")
        
        # Find strong correlations (excluding diagonal)
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_corrs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })
        
        if strong_corrs:
            corr_df = pd.DataFrame(strong_corrs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(
                corr_df.style.format({'Correlation': '{:.3f}'}),
                use_container_width=True
            )
        else:
            st.info("No strong correlations found between variables")
    
    def render_interactive_filters(self, data: pd.DataFrame, key_prefix: str = ""):
        """Render interactive filters for data exploration"""
        if data.empty:
            return {}
        
        filters = {}
        
        st.subheader("🔧 Interactive Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'source' in data.columns:
                available_sources = data['source'].unique()
                selected_sources = st.multiselect(
                    "Sources",
                    available_sources,
                    default=list(available_sources),
                    key=f"{key_prefix}_sources"
                )
                filters['sources'] = selected_sources
        
        with col2:
            if 'credibility_score' in data.columns:
                cred_range = st.slider(
                    "Credibility Range",
                    min_value=float(data['credibility_score'].min()),
                    max_value=float(data['credibility_score'].max()),
                    value=(float(data['credibility_score'].min()), 
                          float(data['credibility_score'].max())),
                    key=f"{key_prefix}_credibility"
                )
                filters['credibility_range'] = cred_range
        
        with col3:
            if 'bias_score' in data.columns:
                bias_range = st.slider(
                    "Bias Score Range",
                    min_value=float(data['bias_score'].min()),
                    max_value=float(data['bias_score'].max()),
                    value=(float(data['bias_score'].min()), 
                          float(data['bias_score'].max())),
                    key=f"{key_prefix}_bias"
                )
                filters['bias_range'] = bias_range
        
        # Date range filter if applicable
        if 'published_at' in data.columns:
            date_col = pd.to_datetime(data['published_at'])
            date_range = st.date_input(
                "Date Range",
                value=(date_col.min().date(), date_col.max().date()),
                min_value=date_col.min().date(),
                max_value=date_col.max().date(),
                key=f"{key_prefix}_dates"
            )
            filters['date_range'] = date_range
        
        return filters
    
    def apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_data = data.copy()
        
        if 'sources' in filters and filters['sources']:
            filtered_data = filtered_data[filtered_data['source'].isin(filters['sources'])]
        
        if 'credibility_range' in filters:
            cred_min, cred_max = filters['credibility_range']
            filtered_data = filtered_data[
                (filtered_data['credibility_score'] >= cred_min) &
                (filtered_data['credibility_score'] <= cred_max)
            ]
        
        if 'bias_range' in filters:
            bias_min, bias_max = filters['bias_range']
            filtered_data = filtered_data[
                (filtered_data['bias_score'] >= bias_min) &
                (filtered_data['bias_score'] <= bias_max)
            ]
        
        if 'date_range' in filters and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            filtered_data['published_at'] = pd.to_datetime(filtered_data['published_at'])
            filtered_data = filtered_data[
                (filtered_data['published_at'].dt.date >= start_date) &
                (filtered_data['published_at'].dt.date <= end_date)
            ]
        
        return filtered_data
    
    def render_summary_stats(self, data: pd.DataFrame, title: str = "Summary Statistics"):
        """Render summary statistics table"""
        if data.empty:
            return
        
        st.subheader(f"📊 {title}")
        
        # Select numeric columns for statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.info("No numeric data available for statistics")
            return
        
        # Calculate summary statistics
        stats = data[numeric_cols].describe()
        
        # Add additional statistics
        stats.loc['median'] = data[numeric_cols].median()
        stats.loc['mode'] = data[numeric_cols].mode().iloc[0] if not data[numeric_cols].mode().empty else np.nan
        
        # Reorder rows for better presentation
        desired_order = ['count', 'mean', 'median', 'mode', 'std', 'min', '25%', '50%', '75%', 'max']
        stats = stats.reindex([idx for idx in desired_order if idx in stats.index])
        
        # Format and display
        formatted_stats = stats.round(3)
        st.dataframe(formatted_stats, use_container_width=True)
        
        # Highlight key insights
        insights = []
        for col in numeric_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                cv = std_val / mean_val  # Coefficient of variation
                if cv > 0.5:
                    insights.append(f"High variability in {col} (CV: {cv:.2f})")
        
        if insights:
            st.subheader("🔍 Key Insights")
            for insight in insights[:3]:  # Show top 3 insights
                st.info(insight)