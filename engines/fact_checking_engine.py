"""
Fact-checking engine using multiple strategies and cross-referencing
"""
import re
# import nltk
import numpy as np
from typing import List, Dict, Optional
# from datetime import datetime, timedelta

from models.data_models import FactCheckResult, NewsArticle
from utils.text_processing import TextProcessor
from config import FACT_CHECK_INDICATORS, MISINFORMATION_PATTERNS, HIGH_SIMILARITY_THRESHOLD, MODERATE_SIMILARITY_THRESHOLD


class FactCheckingEngine:
    """Fact-checking engine using multiple strategies"""
    
    def __init__(self, vector_db, embedding_model):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.text_processor = TextProcessor()
        
        # Fact-checking patterns and indicators
        self.fact_check_indicators = FACT_CHECK_INDICATORS
        self.misinfo_patterns = MISINFORMATION_PATTERNS
        
        # Statistical claim patterns
        self.statistical_patterns = [
            r'\d+%',  # Percentages
            r'\d+\s*percent',  # Percent written out
            r'\d+(?:\.\d+)?\s*(?:times|fold|x)',  # Multipliers
            r'statistics show|data reveals|study found|research indicates',
            r'according to (?:a )?(?:study|report|survey|poll)',
            r'\d+(?:,\d{3})*\s*(?:people|deaths|cases|patients)'
        ]
        
        # Credibility indicators
        self.credibility_indicators = {
            'positive': [
                'peer-reviewed', 'published in', 'according to experts',
                'scientific study', 'clinical trial', 'meta-analysis',
                'university research', 'government data', 'official statistics'
            ],
            'negative': [
                'rumored', 'allegedly', 'claims without evidence',
                'unverified', 'social media post', 'anonymous source',
                'viral video', 'forwarded message'
            ]
        }
    
    def fact_check_article(self, article: NewsArticle) -> List[FactCheckResult]:
        """Perform comprehensive fact-checking on an article"""
        fact_check_results = []
        
        # Extract factual claims
        claims = self.extract_claims(article.content, article.title)
        
        if not claims:
            return fact_check_results
        
        # Check each claim
        for claim in claims:
            result = self._fact_check_single_claim(claim, article)
            if result:
                fact_check_results.append(result)
        
        return fact_check_results
    
    def extract_claims(self, text: str, title: str = "") -> List[str]:
        """Extract factual claims from text"""
        claims = []
        full_text = f"{title}. {text}" if title else text
        
        # Split into sentences
        sentences = self.text_processor.extract_sentences(full_text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for factual indicators
            if any(indicator in sentence_lower for indicator in self.fact_check_indicators):
                claims.append(sentence.strip())
                continue
            
            # Look for statistical claims
            if any(re.search(pattern, sentence_lower) for pattern in self.statistical_patterns):
                claims.append(sentence.strip())
                continue
            
            # Look for specific numeric claims
            if re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', sentence):
                # Check if it's a meaningful claim (not just dates or addresses)
                if not re.search(r'(?:january|february|march|april|may|june|july|august|september|october|november|december)|\d{4}|\d{1,2}:\d{2}', sentence_lower):
                    claims.append(sentence.strip())
        
        # Limit and prioritize claims
        return self._prioritize_claims(claims)[:5]
    
    def _prioritize_claims(self, claims: List[str]) -> List[str]:
        """Prioritize claims based on importance and verifiability"""
        if not claims:
            return claims
        
        scored_claims = []
        
        for claim in claims:
            score = 0
            claim_lower = claim.lower()
            
            # Higher score for statistical claims
            if any(re.search(pattern, claim_lower) for pattern in self.statistical_patterns):
                score += 3
            
            # Higher score for factual indicators
            if any(indicator in claim_lower for indicator in self.fact_check_indicators):
                score += 2
            
            # Higher score for specific numbers
            if re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', claim):
                score += 1
            
            # Lower score for very long sentences (harder to verify)
            if len(claim.split()) > 30:
                score -= 1
            
            # Higher score for credibility indicators
            if any(indicator in claim_lower for indicator in self.credibility_indicators['positive']):
                score += 2
            
            # Lower score for negative credibility indicators
            if any(indicator in claim_lower for indicator in self.credibility_indicators['negative']):
                score -= 2
            
            scored_claims.append((claim, score))
        
        # Sort by score and return top claims
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        return [claim for claim, score in scored_claims]
    
    def _fact_check_single_claim(self, claim: str, article: NewsArticle) -> Optional[FactCheckResult]:
        """Fact-check a single claim"""
        try:
            # Generate embedding for the claim
            claim_embedding = self.embedding_model.encode_text([claim])
            
            # Search for similar content in the database
            similar_articles = self._search_similar_content(claim_embedding, article.id)
            
            # Analyze the evidence
            evidence_analysis = self._analyze_evidence(claim, similar_articles)
            
            # Check for misinformation patterns
            misinfo_score = self.check_misinformation_patterns(claim)
            
            # Determine verdict and confidence
            verdict, confidence, reasoning = self._determine_verdict(
                evidence_analysis, misinfo_score, claim
            )
            
            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence_analysis['supporting_evidence'][:3],
                sources=evidence_analysis['source_list'][:3],
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error fact-checking claim: {str(e)}")
            return None
    
    def _search_similar_content(self, claim_embedding: List[float], exclude_id: str) -> List[Dict]:
        """Search for similar content in the vector database"""
        try:
            results = self.vector_db.query(
                query_embeddings=claim_embedding.tolist(),
                n_results=10
            )
            
            similar_content = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Skip the same article
                    article_id = results['ids'][0][i] if results['ids'] else None
                    if article_id == exclude_id:
                        continue
                    
                    similar_content.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,
                        'article_id': article_id
                    })
            
            return similar_content
            
        except Exception as e:
            print(f"Error searching similar content: {str(e)}")
            return []
    
    def _analyze_evidence(self, claim: str, similar_articles: List[Dict]) -> Dict:
        """Analyze evidence from similar articles"""
        supporting_evidence = []
        contradicting_evidence = []
        source_list = []
        
        for article in similar_articles:
            similarity = article['similarity']
            content = article['content']
            metadata = article['metadata']
            
            # High similarity suggests supporting evidence
            if similarity > 1 - HIGH_SIMILARITY_THRESHOLD:
                supporting_evidence.append(content[:200] + "...")
                if metadata.get('source'):
                    source_list.append(metadata['source'])
            
            # Moderate similarity might be contradicting
            elif similarity > 1 - MODERATE_SIMILARITY_THRESHOLD:
                # Simple keyword contradiction check
                if self._check_contradiction(claim, content):
                    contradicting_evidence.append(content[:200] + "...")
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'source_list': list(set(source_list)),  # Remove duplicates
            'evidence_strength': len(supporting_evidence) - len(contradicting_evidence)
        }
    
    def _check_contradiction(self, claim: str, content: str) -> bool:
        """Check if content contradicts the claim using simple heuristics"""
        claim_lower = claim.lower()
        content_lower = content.lower()
        
        # Look for negation patterns
        negation_patterns = [
            r'not\s+' + re.escape(claim_lower[:20]),
            r'no\s+evidence',
            r'false\s+claim',
            r'debunked',
            r'myth'
        ]
        
        return any(re.search(pattern, content_lower) for pattern in negation_patterns)
    
    def _determine_verdict(self, evidence_analysis: Dict, misinfo_score: float, claim: str) -> tuple:
        """Determine the fact-check verdict and confidence"""
        supporting_count = len(evidence_analysis['supporting_evidence'])
        contradicting_count = len(evidence_analysis['contradicting_evidence'])
        evidence_strength = evidence_analysis['evidence_strength']
        
        # Base confidence on evidence
        base_confidence = min(0.9, max(0.1, (supporting_count + contradicting_count) * 0.15))
        
        # Adjust for misinformation patterns
        if misinfo_score > 0.5:
            verdict = "Likely False"
            confidence = min(0.8, base_confidence + misinfo_score * 0.3)
            reasoning = f"High misinformation indicators detected. Cross-referenced with {supporting_count} supporting and {contradicting_count} contradicting sources."
        
        elif evidence_strength > 2:
            verdict = "Likely True"
            confidence = min(0.9, base_confidence + evidence_strength * 0.1)
            reasoning = f"Strong supporting evidence from {supporting_count} sources with minimal contradiction."
        
        elif evidence_strength < -1:
            verdict = "Questionable"
            confidence = min(0.8, base_confidence + abs(evidence_strength) * 0.1)
            reasoning = f"Evidence suggests potential inaccuracy. Found {contradicting_count} contradicting sources vs {supporting_count} supporting."
        
        elif supporting_count > 0:
            verdict = "Partially Supported"
            confidence = base_confidence
            reasoning = f"Some supporting evidence found ({supporting_count} sources) but limited cross-validation."
        
        else:
            verdict = "Insufficient Evidence"
            confidence = max(0.1, base_confidence - 0.3)
            reasoning = "Unable to find sufficient evidence to verify this claim."
        
        return verdict, confidence, reasoning
    
    def check_misinformation_patterns(self, text: str) -> float:
        """Check for common misinformation patterns"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        pattern_count = 0
        
        # Check misinformation patterns
        for pattern in self.misinfo_patterns:
            if re.search(pattern, text_lower):
                pattern_count += 1
        
        # Additional red flags
        red_flags = [
            r'big pharma',
            r'they don\'t want you to know',
            r'suppressed by the media',
            r'wake up sheeple',
            r'do your own research',
            r'mainstream media lies',
            r'government conspiracy'
        ]
        
        for flag in red_flags:
            if re.search(flag, text_lower):
                pattern_count += 0.5
        
        # Return suspicion score (0-1)
        return min(1.0, pattern_count * 0.2)
    
    def cross_reference_claims(self, claims: List[str]) -> List[FactCheckResult]:
        """Cross-reference multiple claims (legacy method for compatibility)"""
        results = []
        
        for claim in claims:
            # Create a mock article for the claim
            mock_article = type('MockArticle', (), {
                'id': 'temp_claim_check',
                'content': claim,
                'title': ''
            })()
            
            result = self._fact_check_single_claim(claim, mock_article)
            if result:
                results.append(result)
        
        return results
    
    def validate_statistics(self, text: str) -> Dict[str, any]:
        """Validate statistical claims in text"""
        stats = self.text_processor.extract_numbers_and_stats(text)
        
        validation_results = {
            'total_stats': len(stats),
            'suspicious_stats': [],
            'validation_score': 1.0
        }
        
        for stat in stats:
            # Check for suspicious patterns
            if re.search(r'100%|0%', stat):
                validation_results['suspicious_stats'].append({
                    'stat': stat,
                    'reason': 'Absolute percentages are often misleading'
                })
                validation_results['validation_score'] -= 0.1
            
            # Check for very precise numbers that seem manufactured
            if re.search(r'\d+\.\d{3,}', stat):
                validation_results['suspicious_stats'].append({
                    'stat': stat,
                    'reason': 'Unusually precise numbers may be fabricated'
                })
                validation_results['validation_score'] -= 0.05
        
        validation_results['validation_score'] = max(0.0, validation_results['validation_score'])
        return validation_results
    
    def analyze_source_credibility(self, article: NewsArticle) -> Dict[str, any]:
        """Analyze the credibility of sources mentioned in the article"""
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        full_text = f"{title_lower} {content_lower}"
        
        credibility_score = 0.5  # Base score
        indicators = {
            'positive': 0,
            'negative': 0,
            'details': []
        }
        
        # Check for positive credibility indicators
        for indicator in self.credibility_indicators['positive']:
            if indicator in full_text:
                indicators['positive'] += 1
                indicators['details'].append(f"Found: {indicator}")
                credibility_score += 0.1
        
        # Check for negative credibility indicators
        for indicator in self.credibility_indicators['negative']:
            if indicator in full_text:
                indicators['negative'] += 1
                indicators['details'].append(f"Warning: {indicator}")
                credibility_score -= 0.15
        
        # Normalize score
        credibility_score = max(0.0, min(1.0, credibility_score))
        
        return {
            'credibility_score': credibility_score,
            'positive_indicators': indicators['positive'],
            'negative_indicators': indicators['negative'],
            'details': indicators['details']
        }
    
    def get_fact_check_summary(self, fact_check_results: List[FactCheckResult]) -> Dict[str, any]:
        """Generate a summary of fact-check results"""
        if not fact_check_results:
            return {
                'total_claims': 0,
                'verdicts': {},
                'average_confidence': 0.0,
                'overall_assessment': 'No claims to verify'
            }
        
        verdicts = {}
        confidences = []
        
        for result in fact_check_results:
            verdict = result.verdict
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            confidences.append(result.confidence)
        
        average_confidence = np.mean(confidences)
        
        # Determine overall assessment
        total_claims = len(fact_check_results)
        likely_true = verdicts.get('Likely True', 0)
        likely_false = verdicts.get('Likely False', 0)
        questionable = verdicts.get('Questionable', 0)
        
        if likely_false > total_claims * 0.3:
            overall_assessment = 'Contains questionable claims'
        elif likely_true > total_claims * 0.6:
            overall_assessment = 'Generally factual'
        elif questionable > total_claims * 0.4:
            overall_assessment = 'Mixed reliability'
        else:
            overall_assessment = 'Requires further verification'
        
        return {
            'total_claims': total_claims,
            'verdicts': verdicts,
            'average_confidence': average_confidence,
            'overall_assessment': overall_assessment
        }