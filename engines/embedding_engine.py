"""
Embedding engine for generating and managing text embeddings
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Union, Dict
import streamlit as st

from models.data_models import NewsArticle
from config import EMBEDDING_MODEL_NAME, DEVICE


class EmbeddingEngine:
    """Handles text embeddings for the RAG system"""
    
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.device = DEVICE
        self.model = self._load_model()
        self.model_info = self._get_model_info()
    
    def _load_model(self) -> Optional[SentenceTransformer]:
        """Load the sentence transformer model"""
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
            return None
    
    def _get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model:
            return {'error': 'Model not loaded'}
        
        try:
            return {
                'model_name': self.model_name,
                'max_seq_length': getattr(self.model, 'max_seq_length', 512),
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'device': str(self.model.device)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def encode_text(self, text: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """Generate embeddings for text(s)"""
        if not self.model:
            return None
        
        if not text:
            return None
        
        try:
            # Handle single text or list of texts
            if isinstance(text, str):
                texts = [text]
                single_text = True
            else:
                texts = text
                single_text = False
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            
            # Return single embedding or array based on input
            if single_text:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def encode_article(self, article: NewsArticle, combine_fields: bool = True) -> Optional[List[float]]:
        """Generate embeddings for a news article"""
        if combine_fields:
            # Combine title and content for richer representation
            text = f"{article.title}\n\n{article.content}"
        else:
            text = article.content
        
        embedding = self.encode_text(text)
        
        if embedding is not None:
            return embedding.tolist()
        return None
    
    def encode_articles_batch(self, articles: List[NewsArticle], 
                            batch_size: int = 16) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple articles efficiently"""
        if not articles:
            return []
        
        # Prepare texts
        texts = []
        for article in articles:
            text = f"{article.title}\n\n{article.content}"
            texts.append(text)
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts, batch_size=len(batch_texts))
            
            if batch_embeddings is not None:
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
            else:
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch_texts))
        
        return embeddings
    
    def compute_similarity(self, embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def find_similar_articles(self, query_embedding: List[float], 
                            article_embeddings: List[List[float]],
                            top_k: int = 10) -> List[Dict]:
        """Find most similar articles based on embeddings"""
        if not query_embedding or not article_embeddings:
            return []
        
        similarities = []
        
        for i, article_embedding in enumerate(article_embeddings):
            if article_embedding is None:
                continue
            
            similarity = self.compute_similarity(query_embedding, article_embedding)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def cluster_articles(self, articles: List[NewsArticle], 
                        n_clusters: int = 5) -> Dict[int, List[int]]:
        """Cluster articles based on their embeddings"""
        if not articles or len(articles) < n_clusters:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            
            # Generate embeddings for all articles
            embeddings = self.encode_articles_batch(articles)
            
            # Filter out None embeddings
            valid_embeddings = []
            valid_indices = []
            
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_indices.append(i)
            
            if len(valid_embeddings) < n_clusters:
                return {}
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(valid_embeddings)
            
            # Group articles by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                article_index = valid_indices[i]
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(article_index)
            
            return clusters
            
        except ImportError:
            st.warning("Scikit-learn not available for clustering")
            return {}
        except Exception as e:
            st.error(f"Error clustering articles: {str(e)}")
            return {}
    
    def semantic_search(self, query: str, article_texts: List[str], 
                       top_k: int = 10) -> List[Dict]:
        """Perform semantic search on article texts"""
        if not query or not article_texts:
            return []
        
        # Generate query embedding
        query_embedding = self.encode_text(query)
        if query_embedding is None:
            return []
        
        # Generate embeddings for articles
        article_embeddings = self.encode_text(article_texts)
        if article_embeddings is None:
            return []
        
        # Compute similarities
        results = []
        for i, article_embedding in enumerate(article_embeddings):
            similarity = self.compute_similarity(
                query_embedding.tolist(), 
                article_embedding.tolist()
            )
            
            results.append({
                'index': i,
                'similarity': similarity,
                'text': article_texts[i]
            })
        
        # Sort and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict:
        """Get statistics about a collection of embeddings"""
        if not embeddings:
            return {}
        
        try:
            # Filter out None embeddings
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if not valid_embeddings:
                return {'valid_embeddings': 0}
            
            embeddings_array = np.array(valid_embeddings)
            
            stats = {
                'total_embeddings': len(embeddings),
                'valid_embeddings': len(valid_embeddings),
                'embedding_dimension': embeddings_array.shape[1],
                'mean_norm': np.mean(np.linalg.norm(embeddings_array, axis=1)),
                'std_norm': np.std(np.linalg.norm(embeddings_array, axis=1)),
                'mean_values': np.mean(embeddings_array, axis=0).tolist()[:5],  # First 5 dims
                'std_values': np.std(embeddings_array, axis=0).tolist()[:5]     # First 5 dims
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def reduce_dimensions(self, embeddings: List[List[float]], 
                         n_components: int = 2) -> Optional[np.ndarray]:
        """Reduce embedding dimensions for visualization"""
        if not embeddings:
            return None
        
        try:
            from sklearn.decomposition import PCA
            
            # Filter out None embeddings
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if len(valid_embeddings) < n_components:
                return None
            
            embeddings_array = np.array(valid_embeddings)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings_array)
            
            return reduced_embeddings
            
        except ImportError:
            st.warning("Scikit-learn not available for dimensionality reduction")
            return None
        except Exception as e:
            st.error(f"Error reducing dimensions: {str(e)}")
            return None
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate an embedding"""
        if not embedding:
            return False
        
        try:
            # Check if it's a valid list of numbers
            emb_array = np.array(embedding)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(emb_array)) or np.any(np.isinf(emb_array)):
                return False
            
            # Check dimension
            expected_dim = self.model_info.get('embedding_dimension', 384)
            if len(embedding) != expected_dim:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_model_info_display(self) -> Dict:
        """Get model information for display purposes"""
        info = self.model_info.copy()
        
        if self.model:
            info['status'] = 'Loaded successfully'
            info['memory_usage'] = 'Available'  # Could calculate actual memory usage
        else:
            info['status'] = 'Failed to load'
        
        return info