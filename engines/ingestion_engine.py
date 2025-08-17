"""
News ingestion engine for fetching articles from RSS feeds
"""
import feedparser
import streamlit as st
from datetime import datetime
from typing import List, Optional
# import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models.data_models import NewsArticle
from utils.text_processing import TextProcessor
from config import RSS_FEEDS, MAX_ARTICLES_PER_SOURCE, MAX_CONTENT_LENGTH


class NewsIngestionEngine:
    """Real-time news ingestion from multiple free sources"""
    
    def __init__(self):
        self.rss_feeds = RSS_FEEDS
        self.text_processor = TextProcessor()
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout and headers
        session.timeout = 30
        session.headers.update({
            'User-Agent': 'NewsRAG/1.0 (+https://example.com/contact)'
        })
        
        return session
    
    def fetch_articles_from_rss(self, max_articles: int = 100) -> List[NewsArticle]:
        """Fetch articles from RSS feeds"""
        articles = []
        articles_per_source = min(MAX_ARTICLES_PER_SOURCE, max_articles // len(self.rss_feeds))
        
        for source_name, source_config in self.rss_feeds.items():
            try:
                feed_articles = self._fetch_from_source(
                    source_name, 
                    source_config, 
                    articles_per_source
                )
                articles.extend(feed_articles)
                
            except Exception as e:
                st.warning(f"Error fetching from {source_name}: {str(e)}")
                continue
        
        return articles
    
    def _fetch_from_source(self, source_name: str, source_config: dict, max_articles: int) -> List[NewsArticle]:
        """Fetch articles from a single RSS source"""
        articles = []
        
        try:
            # Add timeout and error handling
            feed_url = source_config['url']
            credibility_score = source_config['credibility']
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                st.warning(f"No entries found in feed: {source_name}")
                return articles
            
            for entry in feed.entries[:max_articles]:
                try:
                    article = self._parse_feed_entry(entry, source_name, credibility_score)
                    if article:
                        articles.append(article)
                        
                except Exception as e:
                    st.warning(f"Error parsing entry from {source_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"Failed to fetch from {source_name}: {str(e)}")
        
        return articles
    
    def _parse_feed_entry(self, entry, source_name: str, credibility_score: float) -> Optional[NewsArticle]:
        """Parse a single RSS feed entry into NewsArticle"""
        try:
            # Create unique ID
            article_id = self.text_processor.generate_article_id(
                entry.title, 
                entry.link
            )
            
            # Parse published date
            published_at = self._parse_published_date(entry)
            
            # Extract and clean content
            content = self._extract_content(entry)
            cleaned_content = self.text_processor.clean_text(content)
            
            # Limit content length
            if len(cleaned_content) > MAX_CONTENT_LENGTH:
                cleaned_content = cleaned_content[:MAX_CONTENT_LENGTH] + "..."
            
            # Create description
            description = self.text_processor.truncate_text(cleaned_content, 200)
            
            # Extract author
            author = getattr(entry, 'author', None)
            if not author and hasattr(entry, 'authors'):
                author = ', '.join([a.name for a in entry.authors if hasattr(a, 'name')])
            
            article = NewsArticle(
                id=article_id,
                title=self.text_processor.clean_text(entry.title),
                content=cleaned_content,
                url=entry.link,
                source=source_name,
                published_at=published_at,
                author=author,
                description=description,
                credibility_score=credibility_score
            )
            
            return article
            
        except Exception as e:
            print(f"Error parsing entry: {str(e)}")
            return None
    
    def _parse_published_date(self, entry) -> datetime:
        """Parse published date from RSS entry"""
        # Try different date fields
        for date_field in ['published_parsed', 'updated_parsed']:
            if hasattr(entry, date_field):
                parsed_time = getattr(entry, date_field)
                if parsed_time:
                    try:
                        return datetime(*parsed_time[:6])
                    except (TypeError, ValueError):
                        continue
        
        # Try string dates
        for date_field in ['published', 'updated']:
            if hasattr(entry, date_field):
                date_str = getattr(entry, date_field)
                if date_str:
                    try:
                        # Try various date formats
                        return self._parse_date_string(date_str)
                    except (ValueError, TypeError):
                        continue
        
        # Fallback to current time
        return datetime.now()
    
    def _parse_date_string(self, date_str: str) -> datetime:
        """Parse date string in various formats"""
        import dateutil.parser
        
        try:
            return dateutil.parser.parse(date_str)
        except Exception:
            # Manual parsing for common RSS date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            raise ValueError(f"Unable to parse date: {date_str}")
    
    def _extract_content(self, entry) -> str:
        """Extract content from RSS entry"""
        # Try different content fields in order of preference
        content_fields = [
            'content',  # Full content
            'summary',  # Summary
            'description',  # Description
            'subtitle'  # Subtitle
        ]
        
        for field in content_fields:
            if hasattr(entry, field):
                field_value = getattr(entry, field)
                
                # Handle different content structures
                if isinstance(field_value, list) and len(field_value) > 0:
                    # Content is in a list (like Atom feeds)
                    content_item = field_value[0]
                    if hasattr(content_item, 'value'):
                        return content_item.value
                    elif isinstance(content_item, dict) and 'value' in content_item:
                        return content_item['value']
                    else:
                        return str(content_item)
                
                elif isinstance(field_value, str):
                    return field_value
                
                elif hasattr(field_value, 'value'):
                    return field_value.value
        
        # Fallback to title if no content found
        return getattr(entry, 'title', '')
    
    def fetch_single_article(self, url: str) -> Optional[NewsArticle]:
        """Fetch a single article by URL (if it's from an RSS feed)"""
        for source_name, source_config in self.rss_feeds.items():
            try:
                feed = feedparser.parse(source_config['url'])
                
                for entry in feed.entries:
                    if entry.link == url:
                        return self._parse_feed_entry(
                            entry, 
                            source_name, 
                            source_config['credibility']
                        )
                        
            except Exception:
                continue
        
        return None
    
    def validate_feed(self, feed_url: str) -> dict:
        """Validate an RSS feed URL"""
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                return {
                    'valid': False,
                    'error': 'No entries found in feed'
                }
            
            return {
                'valid': True,
                'title': getattr(feed.feed, 'title', 'Unknown'),
                'description': getattr(feed.feed, 'description', ''),
                'entries_count': len(feed.entries),
                'last_updated': getattr(feed.feed, 'updated', None)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def get_available_sources(self) -> List[dict]:
        """Get list of available RSS sources"""
        sources = []
        
        for source_name, source_config in self.rss_feeds.items():
            sources.append({
                'name': source_name,
                'url': source_config['url'],
                'credibility': source_config['credibility'],
                'status': 'active'  # Could be enhanced to check actual status
            })
        
        return sources
    
    def add_custom_source(self, name: str, url: str, credibility: float = 0.5) -> bool:
        """Add a custom RSS source (runtime only)"""
        try:
            # Validate the feed first
            validation = self.validate_feed(url)
            
            if not validation['valid']:
                return False
            
            # Add to runtime sources
            self.rss_feeds[name] = {
                'url': url,
                'credibility': credibility
            }
            
            return True
            
        except Exception:
            return False