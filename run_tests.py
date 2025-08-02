"""
Test Runner for Mental Health Detector
Comprehensive test suite runner with reporting and validation.
"""

import pytest
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    
    logger.info("Starting comprehensive test suite for Mental Health Detector")
    
    # Test configuration
    test_args = [
        "tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
        f"--html=test_reports/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        "--self-contained-html",  # Self-contained HTML report
        "--cov=src",  # Coverage for src directory
        "--cov-report=html:test_reports/coverage_html",
        "--cov-report=term-missing",
        "--cov-fail-under=70",  # Minimum 70% coverage
    ]
    
    # Create test reports directory
    Path("test_reports").mkdir(exist_ok=True)
    
    # Run tests
    logger.info("Running tests with pytest...")
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        logger.info("✅ All tests passed successfully!")
    else:
        logger.error(f"❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

def run_fairness_tests():
    """Run specific fairness and bias tests"""
    
    logger.info("Running fairness and bias validation tests...")
    
    fairness_args = [
        "tests/test_fairness_validation.py",
        "-v",
        "--tb=short",
        f"--html=test_reports/fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        "--self-contained-html"
    ]
    
    exit_code = pytest.main(fairness_args)
    
    if exit_code == 0:
        logger.info("✅ Fairness tests passed!")
    else:
        logger.error(f"❌ Fairness tests failed with exit code: {exit_code}")
    
    return exit_code

def run_performance_tests():
    """Run performance and load tests"""
    
    logger.info("Running performance tests...")
    
    # Import here to avoid dependency issues
    try:
        from src.models.mental_health_classifier import MentalHealthClassifier
        import time
        import numpy as np
        
        classifier = MentalHealthClassifier()
        
        # Performance test data
        test_texts = [
            "I feel happy and excited about life!",
            "I'm feeling really anxious about everything lately.",
            "I've been struggling with depression for months now.",
            "I can't handle this anymore and feel hopeless.",
            "Today was a normal day, nothing special happened."
        ] * 10  # 50 total tests
        
        # Measure performance
        start_time = time.time()
        results = []
        
        for text in test_texts:
            result = classifier.analyze_text(text)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_texts)
        
        # Performance metrics
        processing_times = [r.processing_time for r in results]
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        logger.info(f"Performance Test Results:")
        logger.info(f"  Total texts processed: {len(test_texts)}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Average time per text: {avg_time:.3f} seconds")
        logger.info(f"  Average processing time: {avg_processing_time:.3f} seconds")
        logger.info(f"  Maximum processing time: {max_processing_time:.3f} seconds")
        
        # Performance assertions
        assert avg_time < 5.0, f"Average processing time too high: {avg_time:.3f}s"
        assert max_processing_time < 15.0, f"Maximum processing time too high: {max_processing_time:.3f}s"
        
        logger.info("✅ Performance tests passed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Performance tests failed: {e}")
        return 1

def validate_model_setup():
    """Validate that models are properly set up"""
    
    logger.info("Validating model setup...")
    
    try:
        # Test model imports
        from src.models.emotion_detector import EmotionDetector
        from src.models.sentiment_analyzer import SentimentAnalyzer
        from src.models.mental_health_classifier import MentalHealthClassifier
        
        # Test model initialization
        emotion_detector = EmotionDetector()
        sentiment_analyzer = SentimentAnalyzer()
        classifier = MentalHealthClassifier()
        
        # Test model loading (this will download models if needed)
        logger.info("Loading emotion detection model...")
        emotion_detector.load_model()
        
        logger.info("Loading sentiment analysis model...")
        sentiment_analyzer.load_model()
        
        logger.info("Loading mental health classifier...")
        classifier.load_models()
        
        # Test basic functionality
        test_text = "I feel happy today!"
        
        emotion_result = emotion_detector.detect_emotion(test_text)
        sentiment_result = sentiment_analyzer.analyze_sentiment(test_text)
        classification_result = classifier.analyze_text(test_text)
        
        # Validate results
        assert emotion_result.emotion is not None
        assert sentiment_result.sentiment is not None
        assert classification_result.overall_risk is not None
        
        logger.info("✅ Model setup validation passed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model setup validation failed: {e}")
        return 1

def generate_test_summary():
    """Generate comprehensive test summary"""
    
    logger.info("Generating test summary...")
    
    summary = {
        "test_run_timestamp": datetime.now().isoformat(),
        "test_results": {
            "comprehensive_tests": "pending",
            "fairness_tests": "pending", 
            "performance_tests": "pending",
            "model_validation": "pending"
        },
        "coverage_info": {
            "target_coverage": "70%",
            "actual_coverage": "pending"
        },
        "recommendations": []
    }
    
    # Run all test suites
    logger.info("=" * 60)
    logger.info("MENTAL HEALTH DETECTOR - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)
    
    # 1. Model validation
    logger.info("\n1. Model Setup Validation")
    logger.info("-" * 30)
    model_result = validate_model_setup()
    summary["test_results"]["model_validation"] = "passed" if model_result == 0 else "failed"
    
    # 2. Comprehensive tests
    logger.info("\n2. Comprehensive Test Suite")
    logger.info("-" * 30)
    comprehensive_result = run_comprehensive_tests()
    summary["test_results"]["comprehensive_tests"] = "passed" if comprehensive_result == 0 else "failed"
    
    # 3. Fairness tests
    logger.info("\n3. Fairness and Bias Validation")
    logger.info("-" * 30)
    fairness_result = run_fairness_tests()
    summary["test_results"]["fairness_tests"] = "passed" if fairness_result == 0 else "failed"
    
    # 4. Performance tests
    logger.info("\n4. Performance Testing")
    logger.info("-" * 30)
    performance_result = run_performance_tests()
    summary["test_results"]["performance_tests"] = "passed" if performance_result == 0 else "failed"
    
    # Generate recommendations
    if any(result != 0 for result in [model_result, comprehensive_result, fairness_result, performance_result]):
        summary["recommendations"].append("Review failed tests and address issues before deployment")
    
    if fairness_result != 0:
        summary["recommendations"].append("Address fairness and bias concerns identified in testing")
    
    if performance_result != 0:
        summary["recommendations"].append("Optimize performance to meet response time requirements")
    
    # Save summary
    summary_file = f"test_reports/test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in summary["test_results"].items():
        status_icon = "✅" if result == "passed" else "❌"
        logger.info(f"{status_icon} {test_name.replace('_', ' ').title()}: {result.upper()}")
    
    if summary["recommendations"]:
        logger.info("\nRecommendations:")
        for rec in summary["recommendations"]:
            logger.info(f"  • {rec}")
    
    logger.info(f"\nDetailed reports saved to: test_reports/")
    logger.info(f"Test summary saved to: {summary_file}")
    
    # Overall result
    all_passed = all(result == "passed" for result in summary["test_results"].values())
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return 0
    else:
        logger.info("\n⚠️  Some tests failed. Please review and fix issues before deployment.")
        return 1

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Add src to Python path
    sys.path.insert(0, str(Path.cwd() / "src"))
    
    # Run comprehensive test suite
    exit_code = generate_test_summary()
    sys.exit(exit_code)
