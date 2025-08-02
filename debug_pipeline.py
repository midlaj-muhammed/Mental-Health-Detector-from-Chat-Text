#!/usr/bin/env python3
"""
Debug script to understand pipeline output format
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from transformers import pipeline

def debug_emotion_pipeline():
    print("🔍 Debugging emotion pipeline...")
    
    # Create emotion pipeline
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    
    # Test with sample text
    text = "I feel happy today!"
    results = emotion_pipeline(text)
    
    print(f"Pipeline results type: {type(results)}")
    print(f"Pipeline results: {results}")
    
    if isinstance(results, list) and len(results) > 0:
        print(f"First result type: {type(results[0])}")
        print(f"First result: {results[0]}")
        
        if isinstance(results[0], list):
            print("Results is a list of lists")
            for i, result in enumerate(results[0]):
                print(f"  Result {i}: {result}")
        elif isinstance(results[0], dict):
            print("Results is a list of dicts")
            for result in results:
                print(f"  Label: {result.get('label')}, Score: {result.get('score')}")

def debug_sentiment_pipeline():
    print("\n🔍 Debugging sentiment pipeline...")
    
    # Create sentiment pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )
    
    # Test with sample text
    text = "I feel happy today!"
    results = sentiment_pipeline(text)
    
    print(f"Pipeline results type: {type(results)}")
    print(f"Pipeline results: {results}")
    
    if isinstance(results, list) and len(results) > 0:
        print(f"First result type: {type(results[0])}")
        print(f"First result: {results[0]}")
        
        if isinstance(results[0], list):
            print("Results is a list of lists")
            for i, result in enumerate(results[0]):
                print(f"  Result {i}: {result}")
        elif isinstance(results[0], dict):
            print("Results is a list of dicts")
            for result in results:
                print(f"  Label: {result.get('label')}, Score: {result.get('score')}")

if __name__ == "__main__":
    debug_emotion_pipeline()
    debug_sentiment_pipeline()
