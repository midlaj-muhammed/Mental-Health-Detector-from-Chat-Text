"""
Sentiment Analysis Model for Mental Health Detection
Advanced sentiment analysis specifically tuned for mental health indicators.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import re

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    polarity_score: float  # -1 to 1
    mental_health_indicators: List[str]
    severity_level: str  # 'mild', 'moderate', 'severe'

class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for mental health screening.
    Combines multiple approaches for comprehensive analysis.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mental health keyword patterns
        self.depression_keywords = {
            'severe': ['suicidal', 'kill myself', 'end it all', 'no point', 'worthless', 'hopeless'],
            'moderate': ['depressed', 'sad', 'empty', 'numb', 'tired', 'exhausted', 'lonely'],
            'mild': ['down', 'blue', 'low', 'unmotivated', 'disconnected']
        }
        
        self.anxiety_keywords = {
            'severe': ['panic attack', 'can\'t breathe', 'terrified', 'overwhelming fear'],
            'moderate': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'afraid'],
            'mild': ['concerned', 'uneasy', 'tense', 'restless']
        }
        
        # Positive mental health indicators
        self.positive_keywords = [
            'happy', 'joy', 'excited', 'grateful', 'hopeful', 'confident',
            'peaceful', 'content', 'optimistic', 'proud', 'accomplished'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> None:
        """Load the sentiment analysis model"""
        try:
            self.logger.info(f"Loading sentiment model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Return all scores (replaces deprecated return_all_scores)
            )
            
            self.logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text while preserving sentiment indicators"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve important punctuation for sentiment
        # Remove excessive punctuation but keep emotional indicators
        text = re.sub(r'[!]{3,}', '!!!', text)
        text = re.sub(r'[?]{3,}', '???', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Truncate if too long
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment with mental health focus
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with comprehensive analysis
        """
        if not self.sentiment_pipeline:
            self.load_model()
        
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return SentimentResult(
                sentiment="neutral",
                confidence=0.0,
                polarity_score=0.0,
                mental_health_indicators=[],
                severity_level="mild"
            )
        
        try:
            # Get sentiment predictions
            results = self.sentiment_pipeline(processed_text)

            # Handle pipeline output format (list of lists)
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]  # Extract the inner list

            # Process results
            sentiment_scores = {result['label'].lower(): result['score'] for result in results}
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            sentiment_label, confidence = primary_sentiment
            
            # Calculate polarity score (-1 to 1)
            polarity_score = self._calculate_polarity_score(sentiment_scores)
            
            # Identify mental health indicators
            indicators = self._identify_mental_health_indicators(processed_text)
            
            # Determine severity level
            severity = self._assess_severity(processed_text, polarity_score, indicators)
            
            return SentimentResult(
                sentiment=sentiment_label,
                confidence=confidence,
                polarity_score=polarity_score,
                mental_health_indicators=indicators,
                severity_level=severity
            )
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return SentimentResult(
                sentiment="error",
                confidence=0.0,
                polarity_score=0.0,
                mental_health_indicators=["analysis_error"],
                severity_level="unknown"
            )
    
    def _calculate_polarity_score(self, sentiment_scores: Dict[str, float]) -> float:
        """Calculate polarity score from sentiment probabilities"""
        positive_score = sentiment_scores.get('positive', 0.0)
        negative_score = sentiment_scores.get('negative', 0.0)
        neutral_score = sentiment_scores.get('neutral', 0.0)
        
        # Convert to polarity scale (-1 to 1)
        polarity = positive_score - negative_score
        
        # Adjust for neutral sentiment
        polarity = polarity * (1 - neutral_score)
        
        return np.clip(polarity, -1.0, 1.0)
    
    def _identify_mental_health_indicators(self, text: str) -> List[str]:
        """Identify mental health indicators in text"""
        indicators = []
        text_lower = text.lower()
        
        # Check for depression indicators
        for severity, keywords in self.depression_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    indicators.append(f"depression_{severity}")
                    break
        
        # Check for anxiety indicators
        for severity, keywords in self.anxiety_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    indicators.append(f"anxiety_{severity}")
                    break
        
        # Check for positive indicators
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        if positive_count > 0:
            indicators.append("positive_mood")
        
        # Additional pattern matching
        if re.search(r'\b(can\'t|cannot)\s+(sleep|eat|focus|concentrate)\b', text_lower):
            indicators.append("functional_impairment")
        
        if re.search(r'\b(always|never|everything|nothing)\b.*\b(wrong|bad|terrible)\b', text_lower):
            indicators.append("cognitive_distortion")
        
        return list(set(indicators))  # Remove duplicates
    
    def _assess_severity(self, text: str, polarity_score: float, indicators: List[str]) -> str:
        """Assess severity level based on multiple factors"""
        
        # Check for severe indicators
        severe_indicators = [ind for ind in indicators if 'severe' in ind]
        if severe_indicators or polarity_score < -0.8:
            return "severe"
        
        # Check for moderate indicators
        moderate_indicators = [ind for ind in indicators if 'moderate' in ind]
        if moderate_indicators or polarity_score < -0.5:
            return "moderate"
        
        # Check for functional impairment
        if "functional_impairment" in indicators or "cognitive_distortion" in indicators:
            return "moderate"
        
        return "mild"
    
    def batch_analyze_sentiment(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_mental_health_summary(self, results: List[SentimentResult]) -> Dict:
        """Generate summary of mental health indicators across multiple texts"""
        if not results:
            return {}
        
        # Aggregate indicators
        all_indicators = []
        severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        polarity_scores = []
        
        for result in results:
            all_indicators.extend(result.mental_health_indicators)
            severity_counts[result.severity_level] += 1
            polarity_scores.append(result.polarity_score)
        
        # Calculate averages and trends
        avg_polarity = np.mean(polarity_scores) if polarity_scores else 0.0
        most_common_indicators = list(set(all_indicators))
        
        return {
            "average_polarity": avg_polarity,
            "severity_distribution": severity_counts,
            "common_indicators": most_common_indicators,
            "total_analyses": len(results),
            "overall_risk_level": max(severity_counts.items(), key=lambda x: x[1])[0]
        }
