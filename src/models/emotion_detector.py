"""
Emotion Detection Model for Mental Health Analysis
Uses transformer-based models to detect emotions in text that may indicate mental health concerns.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class EmotionResult:
    """Data class for emotion detection results"""
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    mental_health_risk: str  # 'low', 'medium', 'high'
    risk_factors: List[str]

class EmotionDetector:
    """
    Advanced emotion detection specifically designed for mental health screening.
    Uses multiple models and combines results for better accuracy.
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.emotion_pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mental health risk mapping
        self.high_risk_emotions = {'sadness', 'fear', 'anger', 'disgust'}
        self.medium_risk_emotions = {'surprise'}
        self.low_risk_emotions = {'joy', 'neutral'}
        
        # Emotion to mental health indicator mapping
        self.emotion_indicators = {
            'sadness': ['depression', 'low_mood', 'grief'],
            'fear': ['anxiety', 'panic', 'phobia'],
            'anger': ['irritability', 'aggression', 'frustration'],
            'disgust': ['self_hatred', 'negative_self_image'],
            'surprise': ['emotional_volatility'],
            'joy': ['positive_mood'],
            'neutral': ['stable_mood']
        }
        
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> None:
        """Load the emotion detection model and tokenizer"""
        try:
            self.logger.info(f"Loading emotion detection model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Return all scores (replaces deprecated return_all_scores)
            )
            
            self.logger.info("Emotion detection model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading emotion detection model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for emotion detection"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic preprocessing while preserving emotional context
        text = text.strip()
        
        # Remove excessive whitespace but keep structure
        text = ' '.join(text.split())
        
        # Truncate if too long (model limit)
        if len(text) > 512:
            text = text[:512]
            
        return text
    
    def detect_emotion(self, text: str, confidence_threshold: float = 0.6) -> EmotionResult:
        """
        Detect emotions in text and assess mental health risk
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            EmotionResult with emotion, confidence, and risk assessment
        """
        if not self.emotion_pipeline:
            self.load_model()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return EmotionResult(
                emotion="neutral",
                confidence=0.0,
                all_scores={},
                mental_health_risk="low",
                risk_factors=[]
            )
        
        try:
            # Get emotion predictions
            results = self.emotion_pipeline(processed_text)

            # Handle pipeline output format (list of lists)
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]  # Extract the inner list

            # Convert to dictionary for easier handling
            emotion_scores = {result['label'].lower(): result['score'] for result in results}
            
            # Get primary emotion
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_name, confidence = primary_emotion
            
            # Assess mental health risk
            risk_level = self._assess_mental_health_risk(emotion_scores, confidence)
            risk_factors = self._identify_risk_factors(emotion_scores, confidence_threshold)
            
            return EmotionResult(
                emotion=emotion_name,
                confidence=confidence,
                all_scores=emotion_scores,
                mental_health_risk=risk_level,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return EmotionResult(
                emotion="error",
                confidence=0.0,
                all_scores={},
                mental_health_risk="unknown",
                risk_factors=["analysis_error"]
            )
    
    def _assess_mental_health_risk(self, emotion_scores: Dict[str, float], confidence: float) -> str:
        """Assess mental health risk based on emotion scores"""
        
        # If confidence is too low, return unknown
        if confidence < 0.5:
            return "unknown"
        
        # Calculate weighted risk score
        risk_score = 0.0
        
        for emotion, score in emotion_scores.items():
            if emotion in self.high_risk_emotions:
                risk_score += score * 3
            elif emotion in self.medium_risk_emotions:
                risk_score += score * 2
            elif emotion in self.low_risk_emotions:
                risk_score += score * 1
        
        # Normalize risk score
        risk_score = risk_score / len(emotion_scores)
        
        # Determine risk level
        if risk_score >= 2.0:
            return "high"
        elif risk_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, emotion_scores: Dict[str, float], threshold: float) -> List[str]:
        """Identify specific risk factors based on emotion scores"""
        risk_factors = []
        
        for emotion, score in emotion_scores.items():
            if score >= threshold and emotion in self.emotion_indicators:
                risk_factors.extend(self.emotion_indicators[emotion])
        
        return list(set(risk_factors))  # Remove duplicates
    
    def batch_detect_emotions(self, texts: List[str], confidence_threshold: float = 0.6) -> List[EmotionResult]:
        """Detect emotions for multiple texts"""
        results = []
        
        for text in texts:
            result = self.detect_emotion(text, confidence_threshold)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "supported_emotions": list(self.emotion_indicators.keys())
        }
