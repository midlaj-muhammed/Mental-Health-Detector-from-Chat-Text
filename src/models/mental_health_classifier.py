"""
Mental Health Classifier
Combines emotion detection and sentiment analysis for comprehensive mental health screening.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import json

from .emotion_detector import EmotionDetector, EmotionResult
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult

@dataclass
class MentalHealthResult:
    """Comprehensive mental health analysis result"""
    text: str
    timestamp: datetime
    
    # Primary classifications
    anxiety_risk: str  # 'low', 'medium', 'high'
    depression_risk: str  # 'low', 'medium', 'high'
    overall_risk: str  # 'low', 'medium', 'high', 'crisis'
    
    # Confidence scores
    anxiety_confidence: float
    depression_confidence: float
    overall_confidence: float
    
    # Detailed analysis
    emotion_result: EmotionResult
    sentiment_result: SentimentResult
    
    # Risk factors and indicators
    risk_factors: List[str]
    protective_factors: List[str]
    
    # Recommendations
    recommendations: List[str]
    crisis_indicators: bool
    
    # Metadata
    model_version: str
    processing_time: float

class MentalHealthClassifier:
    """
    Comprehensive mental health classifier that combines multiple models
    for accurate and responsible mental health screening.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.emotion_detector = EmotionDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Risk thresholds - made more nuanced
        self.anxiety_threshold = self.config.get('anxiety_threshold', 0.75)
        self.depression_threshold = self.config.get('depression_threshold', 0.75)
        self.crisis_threshold = self.config.get('crisis_threshold', 0.90)
        
        # Crisis keywords (immediate intervention needed)
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'not worth living', 'better off dead', 'end it all',
            'hurt myself', 'self harm', 'cut myself'
        ]
        
        # Protective factors
        self.protective_keywords = [
            'support', 'family', 'friends', 'therapy', 'counseling',
            'medication', 'exercise', 'hobbies', 'goals', 'future',
            'hope', 'help', 'better', 'improving', 'grateful'
        ]
        
        self.logger = logging.getLogger(__name__)
        self.model_version = "1.0.0"
    
    def load_models(self) -> None:
        """Load all required models"""
        self.logger.info("Loading mental health classification models...")
        self.emotion_detector.load_model()
        self.sentiment_analyzer.load_model()
        self.logger.info("All models loaded successfully")
    
    def analyze_text(self, text: str) -> MentalHealthResult:
        """
        Perform comprehensive mental health analysis on text
        
        Args:
            text: Input text to analyze
            
        Returns:
            MentalHealthResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        if not text or not isinstance(text, str):
            return self._create_empty_result(text, start_time)
        
        try:
            # Perform emotion detection
            emotion_result = self.emotion_detector.detect_emotion(text)
            
            # Perform sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(text)
            
            # Check for crisis indicators
            crisis_indicators = self._check_crisis_indicators(text)
            
            # Calculate risk levels
            anxiety_risk, anxiety_confidence = self._calculate_anxiety_risk(
                emotion_result, sentiment_result, text
            )
            
            depression_risk, depression_confidence = self._calculate_depression_risk(
                emotion_result, sentiment_result, text
            )
            
            overall_risk, overall_confidence = self._calculate_overall_risk(
                anxiety_risk, depression_risk, anxiety_confidence, 
                depression_confidence, crisis_indicators
            )
            
            # Identify risk and protective factors
            risk_factors = self._identify_risk_factors(emotion_result, sentiment_result, text)
            protective_factors = self._identify_protective_factors(text)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_risk, anxiety_risk, depression_risk, crisis_indicators
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MentalHealthResult(
                text=text,
                timestamp=start_time,
                anxiety_risk=anxiety_risk,
                depression_risk=depression_risk,
                overall_risk=overall_risk,
                anxiety_confidence=anxiety_confidence,
                depression_confidence=depression_confidence,
                overall_confidence=overall_confidence,
                emotion_result=emotion_result,
                sentiment_result=sentiment_result,
                risk_factors=risk_factors,
                protective_factors=protective_factors,
                recommendations=recommendations,
                crisis_indicators=crisis_indicators,
                model_version=self.model_version,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in mental health analysis: {str(e)}")
            return self._create_error_result(text, start_time, str(e))
    
    def _check_crisis_indicators(self, text: str) -> bool:
        """Check for immediate crisis indicators"""
        text_lower = text.lower()
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _calculate_anxiety_risk(self, emotion_result: EmotionResult, 
                               sentiment_result: SentimentResult, text: str) -> Tuple[str, float]:
        """Calculate anxiety risk level and confidence"""
        
        # Base score from emotion detection
        anxiety_score = 0.0
        
        # Emotion-based scoring
        if emotion_result.emotion in ['fear', 'surprise']:
            anxiety_score += emotion_result.confidence * 0.4
        
        # Sentiment-based scoring
        anxiety_indicators = [ind for ind in sentiment_result.mental_health_indicators 
                            if 'anxiety' in ind]
        if anxiety_indicators:
            anxiety_score += 0.3
            
        # Severity adjustment
        if sentiment_result.severity_level == 'severe':
            anxiety_score += 0.2
        elif sentiment_result.severity_level == 'moderate':
            anxiety_score += 0.1
        
        # Text pattern analysis
        anxiety_patterns = ['worried', 'anxious', 'panic', 'nervous', 'stress']
        pattern_matches = sum(1 for pattern in anxiety_patterns if pattern in text.lower())
        anxiety_score += min(pattern_matches * 0.05, 0.2)
        
        # Determine risk level with more gradual thresholds
        if anxiety_score >= 0.85:
            return "high", min(anxiety_score, 1.0)
        elif anxiety_score >= 0.60:
            return "medium", min(anxiety_score, 1.0)
        elif anxiety_score >= 0.30:
            return "low", min(anxiety_score, 1.0)
        else:
            return "low", max(anxiety_score, 0.1)  # Minimum confidence
    
    def _calculate_depression_risk(self, emotion_result: EmotionResult, 
                                  sentiment_result: SentimentResult, text: str) -> Tuple[str, float]:
        """Calculate depression risk level and confidence"""
        
        depression_score = 0.0
        
        # Emotion-based scoring
        if emotion_result.emotion in ['sadness', 'disgust']:
            depression_score += emotion_result.confidence * 0.4
        
        # Sentiment-based scoring
        if sentiment_result.polarity_score < -0.3:
            depression_score += abs(sentiment_result.polarity_score) * 0.3
        
        depression_indicators = [ind for ind in sentiment_result.mental_health_indicators 
                               if 'depression' in ind]
        if depression_indicators:
            depression_score += 0.3
        
        # Severity adjustment
        if sentiment_result.severity_level == 'severe':
            depression_score += 0.2
        elif sentiment_result.severity_level == 'moderate':
            depression_score += 0.1
        
        # Text pattern analysis
        depression_patterns = ['sad', 'hopeless', 'empty', 'worthless', 'tired']
        pattern_matches = sum(1 for pattern in depression_patterns if pattern in text.lower())
        depression_score += min(pattern_matches * 0.05, 0.2)
        
        # Determine risk level with more gradual thresholds
        if depression_score >= 0.85:
            return "high", min(depression_score, 1.0)
        elif depression_score >= 0.60:
            return "medium", min(depression_score, 1.0)
        elif depression_score >= 0.30:
            return "low", min(depression_score, 1.0)
        else:
            return "low", max(depression_score, 0.1)  # Minimum confidence
    
    def _calculate_overall_risk(self, anxiety_risk: str, depression_risk: str,
                               anxiety_conf: float, depression_conf: float,
                               crisis_indicators: bool) -> Tuple[str, float]:
        """Calculate overall mental health risk"""
        
        if crisis_indicators:
            return "crisis", 0.95
        
        # Risk level mapping
        risk_mapping = {"low": 1, "medium": 2, "high": 3}
        
        anxiety_level = risk_mapping[anxiety_risk]
        depression_level = risk_mapping[depression_risk]
        
        # Calculate weighted average
        overall_level = max(anxiety_level, depression_level)
        overall_confidence = max(anxiety_conf, depression_conf)
        
        # Map back to risk levels
        if overall_level >= 3:
            return "high", overall_confidence
        elif overall_level >= 2:
            return "medium", overall_confidence
        else:
            return "low", overall_confidence
    
    def _identify_risk_factors(self, emotion_result: EmotionResult, 
                              sentiment_result: SentimentResult, text: str) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # From emotion analysis
        risk_factors.extend(emotion_result.risk_factors)
        
        # From sentiment analysis
        risk_factors.extend(sentiment_result.mental_health_indicators)
        
        # Additional text-based risk factors
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['isolated', 'alone', 'lonely']):
            risk_factors.append('social_isolation')
        
        if any(word in text_lower for word in ['sleep', 'insomnia', 'tired']):
            risk_factors.append('sleep_disturbance')
        
        if any(word in text_lower for word in ['appetite', 'eating', 'food']):
            risk_factors.append('appetite_changes')
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _identify_protective_factors(self, text: str) -> List[str]:
        """Identify protective factors in text"""
        protective_factors = []
        text_lower = text.lower()
        
        for keyword in self.protective_keywords:
            if keyword in text_lower:
                protective_factors.append(keyword)
        
        return protective_factors
    
    def _generate_recommendations(self, overall_risk: str, anxiety_risk: str, 
                                 depression_risk: str, crisis_indicators: bool) -> List[str]:
        """Generate appropriate recommendations based on risk assessment"""
        recommendations = []
        
        if crisis_indicators:
            recommendations.extend([
                "IMMEDIATE: Contact emergency services (911) or crisis hotline (988)",
                "Go to nearest emergency room",
                "Contact a trusted friend or family member immediately",
                "Do not be alone - seek immediate support"
            ])
            return recommendations
        
        if overall_risk == "high":
            recommendations.extend([
                "Strongly recommend consulting with a mental health professional",
                "Consider contacting your primary care physician",
                "Reach out to trusted friends or family members",
                "Contact a mental health crisis line if needed: 988"
            ])
        
        if overall_risk in ["medium", "high"]:
            recommendations.extend([
                "Consider speaking with a counselor or therapist",
                "Practice self-care activities",
                "Maintain regular sleep and exercise routines",
                "Stay connected with supportive people"
            ])
        
        if anxiety_risk in ["medium", "high"]:
            recommendations.extend([
                "Try relaxation techniques (deep breathing, meditation)",
                "Consider anxiety management strategies",
                "Limit caffeine and alcohol consumption"
            ])
        
        if depression_risk in ["medium", "high"]:
            recommendations.extend([
                "Engage in activities you usually enjoy",
                "Maintain social connections",
                "Consider light therapy or outdoor activities",
                "Monitor mood patterns"
            ])
        
        # General recommendations
        recommendations.extend([
            "Remember: This is not a medical diagnosis",
            "Professional help is available and effective",
            "Your mental health matters"
        ])
        
        return recommendations
    
    def _create_empty_result(self, text: str, start_time: datetime) -> MentalHealthResult:
        """Create empty result for invalid input"""
        return MentalHealthResult(
            text=text or "",
            timestamp=start_time,
            anxiety_risk="low",
            depression_risk="low",
            overall_risk="low",
            anxiety_confidence=0.0,
            depression_confidence=0.0,
            overall_confidence=0.0,
            emotion_result=EmotionResult("neutral", 0.0, {}, "low", []),
            sentiment_result=SentimentResult("neutral", 0.0, 0.0, [], "mild"),
            risk_factors=[],
            protective_factors=[],
            recommendations=["Please provide text for analysis"],
            crisis_indicators=False,
            model_version=self.model_version,
            processing_time=0.0
        )
    
    def _create_error_result(self, text: str, start_time: datetime, error: str) -> MentalHealthResult:
        """Create error result"""
        return MentalHealthResult(
            text=text,
            timestamp=start_time,
            anxiety_risk="unknown",
            depression_risk="unknown",
            overall_risk="unknown",
            anxiety_confidence=0.0,
            depression_confidence=0.0,
            overall_confidence=0.0,
            emotion_result=EmotionResult("error", 0.0, {}, "unknown", ["analysis_error"]),
            sentiment_result=SentimentResult("error", 0.0, 0.0, ["analysis_error"], "unknown"),
            risk_factors=["analysis_error"],
            protective_factors=[],
            recommendations=[f"Analysis error: {error}"],
            crisis_indicators=False,
            model_version=self.model_version,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
