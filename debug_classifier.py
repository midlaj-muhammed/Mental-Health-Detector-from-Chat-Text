#!/usr/bin/env python3
"""
Debug the mental health classifier to find the issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def debug_individual_components():
    """Debug each component individually"""
    print("🔍 Debugging Individual Components")
    print("=" * 50)
    
    # Test emotion detector
    print("\n1. Testing Emotion Detector:")
    from src.models.emotion_detector import EmotionDetector
    
    emotion_detector = EmotionDetector()
    emotion_detector.load_model()
    
    test_text = "I feel really anxious and worried about everything"
    emotion_result = emotion_detector.detect_emotion(test_text)
    
    print(f"   Text: {test_text}")
    print(f"   Emotion: {emotion_result.emotion}")
    print(f"   Confidence: {emotion_result.confidence}")
    print(f"   All scores: {emotion_result.all_scores}")
    print(f"   Risk factors: {emotion_result.risk_factors}")
    
    # Test sentiment analyzer
    print("\n2. Testing Sentiment Analyzer:")
    from src.models.sentiment_analyzer import SentimentAnalyzer
    
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_model()
    
    sentiment_result = sentiment_analyzer.analyze_sentiment(test_text)
    
    print(f"   Text: {test_text}")
    print(f"   Sentiment: {sentiment_result.sentiment}")
    print(f"   Confidence: {sentiment_result.confidence}")
    print(f"   Polarity: {sentiment_result.polarity_score}")
    print(f"   Indicators: {sentiment_result.mental_health_indicators}")
    print(f"   Severity: {sentiment_result.severity_level}")
    
    # Test mental health classifier
    print("\n3. Testing Mental Health Classifier:")
    from src.models.mental_health_classifier import MentalHealthClassifier
    
    classifier = MentalHealthClassifier()
    classifier.load_models()
    
    mh_result = classifier.analyze_text(test_text)
    
    print(f"   Text: {test_text}")
    print(f"   Overall Risk: {mh_result.overall_risk}")
    print(f"   Overall Confidence: {mh_result.overall_confidence}")
    print(f"   Anxiety Risk: {mh_result.anxiety_risk}")
    print(f"   Anxiety Confidence: {mh_result.anxiety_confidence}")
    print(f"   Depression Risk: {mh_result.depression_risk}")
    print(f"   Depression Confidence: {mh_result.depression_confidence}")
    print(f"   Crisis Indicators: {mh_result.crisis_indicators}")
    print(f"   Risk Factors: {mh_result.risk_factors}")
    
    return emotion_result, sentiment_result, mh_result

def debug_risk_calculation():
    """Debug the risk calculation logic"""
    print("\n🔍 Debugging Risk Calculation Logic")
    print("=" * 50)
    
    from src.models.mental_health_classifier import MentalHealthClassifier
    
    classifier = MentalHealthClassifier()
    classifier.load_models()
    
    # Test with a clearly anxious text
    anxious_text = "I'm having panic attacks and feel terrified all the time. I can't stop worrying and my heart races constantly."
    
    print(f"\nTesting with anxious text: {anxious_text}")
    
    # Get individual component results
    emotion_result = classifier.emotion_detector.detect_emotion(anxious_text)
    sentiment_result = classifier.sentiment_analyzer.analyze_sentiment(anxious_text)
    
    print(f"\nEmotion Result:")
    print(f"   Emotion: {emotion_result.emotion}")
    print(f"   Confidence: {emotion_result.confidence}")
    print(f"   All scores: {emotion_result.all_scores}")
    
    print(f"\nSentiment Result:")
    print(f"   Sentiment: {sentiment_result.sentiment}")
    print(f"   Confidence: {sentiment_result.confidence}")
    print(f"   Polarity: {sentiment_result.polarity_score}")
    print(f"   Indicators: {sentiment_result.mental_health_indicators}")
    
    # Test the risk calculation methods directly
    print(f"\nTesting risk calculation methods:")
    
    try:
        anxiety_risk, anxiety_conf = classifier._calculate_anxiety_risk(emotion_result, sentiment_result, anxious_text)
        print(f"   Anxiety Risk: {anxiety_risk} (confidence: {anxiety_conf})")
    except Exception as e:
        print(f"   ❌ Anxiety risk calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        depression_risk, depression_conf = classifier._calculate_depression_risk(emotion_result, sentiment_result, anxious_text)
        print(f"   Depression Risk: {depression_risk} (confidence: {depression_conf})")
    except Exception as e:
        print(f"   ❌ Depression risk calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        emotion_result, sentiment_result, mh_result = debug_individual_components()
        debug_risk_calculation()
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
