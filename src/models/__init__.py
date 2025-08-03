# Mental Health Models Package

"""
Mental Health Detection Models

This package contains AI models for mental health analysis:
- EmotionDetector: Detects emotions in text
- SentimentAnalyzer: Analyzes sentiment and polarity
- MentalHealthClassifier: Classifies mental health risk levels
"""

from .emotion_detector import EmotionDetector
from .sentiment_analyzer import SentimentAnalyzer
from .mental_health_classifier import MentalHealthClassifier

__all__ = [
    'EmotionDetector',
    'SentimentAnalyzer',
    'MentalHealthClassifier'
]
