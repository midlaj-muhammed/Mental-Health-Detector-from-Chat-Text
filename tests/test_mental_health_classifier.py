"""
Test suite for Mental Health Classifier
Comprehensive tests for accuracy, fairness, and edge cases.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.mental_health_classifier import MentalHealthClassifier, MentalHealthResult
from src.models.emotion_detector import EmotionDetector
from src.models.sentiment_analyzer import SentimentAnalyzer

class TestMentalHealthClassifier:
    """Test cases for Mental Health Classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing"""
        return MentalHealthClassifier()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return {
            'positive': [
                "I'm feeling great today! Life is wonderful and I'm excited about the future.",
                "Had an amazing day with friends. Feeling grateful and happy.",
                "Just got promoted at work! I'm so proud of myself and optimistic about what's ahead."
            ],
            'neutral': [
                "Today was an ordinary day. Went to work, came home, watched TV.",
                "The weather is nice today. I might go for a walk later.",
                "I need to buy groceries and do laundry this weekend."
            ],
            'mild_concern': [
                "I've been feeling a bit down lately, but I think it will pass.",
                "Work has been stressful and I'm having trouble sleeping sometimes.",
                "I feel overwhelmed with everything I need to do, but I'm managing."
            ],
            'moderate_concern': [
                "I've been feeling really sad and empty for weeks now. Nothing seems to bring me joy.",
                "I'm constantly worried about everything and can't seem to relax or focus.",
                "I feel like I'm failing at everything and don't know how to cope anymore."
            ],
            'high_concern': [
                "I feel completely hopeless and worthless. I don't see the point in anything anymore.",
                "I can't stop the panic attacks and I'm terrified all the time. I feel like I'm losing my mind.",
                "Everything feels overwhelming and I don't think I can handle this anymore. I feel so alone."
            ],
            'crisis': [
                "I don't want to be here anymore. I can't take this pain.",
                "I've been thinking about ending it all. Nothing matters anymore.",
                "I feel like everyone would be better off without me. I can't go on like this."
            ]
        }
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert classifier.model_version == "1.0.0"
        assert hasattr(classifier, 'emotion_detector')
        assert hasattr(classifier, 'sentiment_analyzer')
    
    def test_positive_text_analysis(self, classifier, sample_texts):
        """Test analysis of positive texts"""
        for text in sample_texts['positive']:
            result = classifier.analyze_text(text)
            
            assert isinstance(result, MentalHealthResult)
            assert result.overall_risk in ['low', 'medium']  # Should not be high for positive text
            assert result.crisis_indicators == False
            assert len(result.protective_factors) > 0  # Should identify some protective factors
    
    def test_crisis_detection(self, classifier, sample_texts):
        """Test crisis indicator detection"""
        for text in sample_texts['crisis']:
            result = classifier.analyze_text(text)
            
            assert result.crisis_indicators == True or result.overall_risk == 'crisis'
            assert 'IMMEDIATE' in ' '.join(result.recommendations)
    
    def test_risk_level_progression(self, classifier, sample_texts):
        """Test that risk levels progress appropriately"""
        risk_levels = []
        
        for category in ['positive', 'neutral', 'mild_concern', 'moderate_concern', 'high_concern']:
            for text in sample_texts[category][:1]:  # Test one from each category
                result = classifier.analyze_text(text)
                risk_levels.append(result.overall_risk)
        
        # Check that risk generally increases (allowing for some variation)
        risk_mapping = {'low': 1, 'medium': 2, 'high': 3, 'crisis': 4}
        risk_scores = [risk_mapping[risk] for risk in risk_levels]
        
        # Should show general upward trend
        assert risk_scores[0] <= risk_scores[2]  # positive <= mild_concern
        assert risk_scores[2] <= risk_scores[4]  # mild_concern <= high_concern
    
    def test_confidence_scores(self, classifier, sample_texts):
        """Test confidence score validity"""
        for category, texts in sample_texts.items():
            for text in texts[:1]:  # Test one from each category
                result = classifier.analyze_text(text)
                
                # Confidence scores should be between 0 and 1
                assert 0 <= result.overall_confidence <= 1
                assert 0 <= result.anxiety_confidence <= 1
                assert 0 <= result.depression_confidence <= 1
    
    def test_empty_text_handling(self, classifier):
        """Test handling of empty or invalid text"""
        test_cases = ["", None, "   ", "a"]  # Empty, None, whitespace, too short
        
        for text in test_cases:
            result = classifier.analyze_text(text)
            
            assert isinstance(result, MentalHealthResult)
            assert result.overall_risk == 'low'  # Should default to low risk
            assert result.overall_confidence == 0.0
    
    def test_long_text_handling(self, classifier):
        """Test handling of very long text"""
        long_text = "I feel sad. " * 1000  # Very long repetitive text
        
        result = classifier.analyze_text(long_text)
        
        assert isinstance(result, MentalHealthResult)
        assert result.processing_time < 30.0  # Should complete within reasonable time
    
    def test_multilingual_robustness(self, classifier):
        """Test robustness with non-English text"""
        non_english_texts = [
            "Je me sens triste aujourd'hui",  # French
            "Me siento muy ansioso",  # Spanish
            "Ich bin sehr glücklich",  # German
        ]
        
        for text in non_english_texts:
            result = classifier.analyze_text(text)
            
            # Should handle gracefully without crashing
            assert isinstance(result, MentalHealthResult)
            # May have lower confidence due to language mismatch
    
    def test_special_characters_handling(self, classifier):
        """Test handling of special characters and emojis"""
        special_texts = [
            "I'm feeling 😢 today... really down 💔",
            "Anxiety through the roof!!! 😰😰😰",
            "Life is good 😊✨🌟",
            "Text with @mentions and #hashtags feeling #depressed",
        ]
        
        for text in special_texts:
            result = classifier.analyze_text(text)
            
            assert isinstance(result, MentalHealthResult)
            # Should extract meaningful sentiment despite special characters
    
    def test_demographic_fairness(self, classifier):
        """Test for demographic fairness in predictions"""
        # Test texts with demographic indicators
        demographic_texts = [
            ("I'm a 20-year-old woman feeling anxious about college", "young_female"),
            ("As a 45-year-old man, I'm struggling with work stress", "middle_aged_male"),
            ("I'm an elderly person feeling lonely and sad", "elderly"),
            ("As a teenager, everything feels overwhelming", "teenager"),
        ]
        
        results = []
        for text, demographic in demographic_texts:
            result = classifier.analyze_text(text)
            results.append((result, demographic))
        
        # Check that similar emotional content gets similar risk assessments
        # regardless of demographic indicators
        anxiety_risks = [r[0].anxiety_risk for r in results]
        
        # Should not have extreme variations based solely on demographics
        assert len(set(anxiety_risks)) <= 3  # Allow some variation but not extreme
    
    def test_consistency_across_similar_texts(self, classifier):
        """Test consistency for similar emotional content"""
        similar_texts = [
            "I feel really sad and hopeless about everything",
            "Everything seems hopeless and I feel very sad",
            "I'm feeling hopeless and really sad about life",
        ]
        
        results = [classifier.analyze_text(text) for text in similar_texts]
        
        # Should have similar risk assessments
        overall_risks = [r.overall_risk for r in results]
        depression_risks = [r.depression_risk for r in results]
        
        # All should be the same or very similar
        assert len(set(overall_risks)) <= 2
        assert len(set(depression_risks)) <= 2
    
    def test_recommendation_quality(self, classifier, sample_texts):
        """Test quality and appropriateness of recommendations"""
        for category, texts in sample_texts.items():
            for text in texts[:1]:
                result = classifier.analyze_text(text)
                
                # Should always have recommendations
                assert len(result.recommendations) > 0
                
                # Crisis texts should have immediate action recommendations
                if category == 'crisis':
                    crisis_keywords = ['IMMEDIATE', 'emergency', '911', '988']
                    has_crisis_rec = any(
                        any(keyword in rec for keyword in crisis_keywords)
                        for rec in result.recommendations
                    )
                    assert has_crisis_rec
                
                # Should include disclaimer
                disclaimer_found = any(
                    'not a medical diagnosis' in rec.lower() or 'professional' in rec.lower()
                    for rec in result.recommendations
                )
                assert disclaimer_found
    
    def test_processing_time_performance(self, classifier, sample_texts):
        """Test processing time performance"""
        processing_times = []
        
        for category, texts in sample_texts.items():
            for text in texts:
                result = classifier.analyze_text(text)
                processing_times.append(result.processing_time)
        
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        
        # Performance benchmarks
        assert avg_time < 5.0  # Average should be under 5 seconds
        assert max_time < 15.0  # No single analysis should take more than 15 seconds
    
    def test_risk_factor_identification(self, classifier):
        """Test identification of specific risk factors"""
        risk_factor_texts = {
            'social_isolation': "I feel so alone and isolated from everyone",
            'sleep_disturbance': "I can't sleep at night and I'm always tired",
            'appetite_changes': "I have no appetite and don't want to eat anything",
            'functional_impairment': "I can't focus on work or do basic tasks anymore",
        }
        
        for expected_factor, text in risk_factor_texts.items():
            result = classifier.analyze_text(text)
            
            # Should identify the specific risk factor
            risk_factors_str = ' '.join(result.risk_factors).lower()
            assert any(
                keyword in risk_factors_str 
                for keyword in expected_factor.split('_')
            )
    
    def test_protective_factor_identification(self, classifier):
        """Test identification of protective factors"""
        protective_texts = [
            "I have great support from my family and friends",
            "I'm going to therapy and it's really helping me",
            "I exercise regularly and it makes me feel better",
            "I'm grateful for the good things in my life",
        ]
        
        for text in protective_texts:
            result = classifier.analyze_text(text)
            
            # Should identify some protective factors
            assert len(result.protective_factors) > 0
    
    def test_model_version_tracking(self, classifier):
        """Test model version tracking"""
        result = classifier.analyze_text("Test text for version tracking")
        
        assert result.model_version is not None
        assert isinstance(result.model_version, str)
        assert len(result.model_version) > 0
    
    def test_timestamp_accuracy(self, classifier):
        """Test timestamp accuracy"""
        before_time = datetime.now()
        result = classifier.analyze_text("Test text for timestamp")
        after_time = datetime.now()
        
        assert before_time <= result.timestamp <= after_time

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def classifier(self):
        return MentalHealthClassifier()
    
    def test_extremely_short_text(self, classifier):
        """Test extremely short text inputs"""
        short_texts = ["a", "I", "no", "ok", "hi"]
        
        for text in short_texts:
            result = classifier.analyze_text(text)
            assert isinstance(result, MentalHealthResult)
    
    def test_repetitive_text(self, classifier):
        """Test highly repetitive text"""
        repetitive_text = "sad " * 100
        
        result = classifier.analyze_text(repetitive_text)
        assert isinstance(result, MentalHealthResult)
        # Should still provide meaningful analysis
        assert result.depression_risk in ['low', 'medium', 'high']
    
    def test_mixed_emotional_content(self, classifier):
        """Test text with mixed emotional signals"""
        mixed_text = """
        I'm really happy about my new job and excited about the future!
        But I'm also feeling anxious about the responsibilities and worried I might fail.
        Sometimes I feel sad when I think about leaving my old colleagues behind.
        Overall though, I think this is a positive change in my life.
        """
        
        result = classifier.analyze_text(mixed_text)
        assert isinstance(result, MentalHealthResult)
        
        # Should handle mixed emotions appropriately
        assert result.overall_confidence > 0.3  # Should have reasonable confidence
    
    def test_clinical_terminology(self, classifier):
        """Test text with clinical/medical terminology"""
        clinical_text = """
        I've been diagnosed with major depressive disorder and generalized anxiety disorder.
        I'm taking SSRIs and going to cognitive behavioral therapy sessions.
        My psychiatrist says I have treatment-resistant depression.
        """
        
        result = classifier.analyze_text(clinical_text)
        assert isinstance(result, MentalHealthResult)
        
        # Should recognize clinical context
        assert result.depression_risk in ['medium', 'high']
        assert result.anxiety_risk in ['medium', 'high']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
