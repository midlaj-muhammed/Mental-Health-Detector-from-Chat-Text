#!/usr/bin/env python3
"""
Mental Health Detector - Command Line Interface

Usage:
    python -m src.cli --text "Your text here"
    python -m src.cli --file path/to/text.txt
    python -m src.cli --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.analysis_engine import AnalysisEngine


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Mental Health Detector - AI-powered text analysis for mental health indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.cli --text "I feel anxious about work lately"
    python -m src.cli --file journal.txt --output results.json
    python -m src.cli --text "I'm feeling great today!" --format table

Crisis Resources:
    National Suicide Prevention Lifeline: 988
    Crisis Text Line: Text HOME to 741741
    Emergency Services: 911

⚠️  IMPORTANT: This tool is NOT a substitute for professional mental health care.
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="Text to analyze directly"
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to text file to analyze"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "table", "summary"],
        default="summary",
        help="Output format (default: summary)"
    )
    
    # Analysis options
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for recommendations (default: 0.5)"
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw model outputs in results"
    )
    
    # Utility options
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Mental Health Detector 1.0.0"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    return parser


def load_text_from_file(file_path: Path) -> str:
    """Load text from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"❌ Error: Unable to decode file '{file_path}'. Please ensure it's UTF-8 encoded.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def format_results_json(results: dict) -> str:
    """Format results as JSON."""
    return json.dumps(results, indent=2, default=str)


def format_results_table(results: dict) -> str:
    """Format results as a table."""
    if results.get('error'):
        return f"❌ Analysis Error: {results.get('error_message', 'Unknown error')}"
    
    mh = results.get('mental_health_analysis', {})
    emotion = results.get('emotion_analysis', {})
    sentiment = results.get('sentiment_analysis', {})
    crisis = results.get('crisis_response', {})
    
    output = []
    output.append("🧠 Mental Health Analysis Results")
    output.append("=" * 50)
    output.append("")
    
    # Risk Assessment
    output.append("📊 Risk Assessment:")
    output.append(f"   Overall Risk: {mh.get('overall_risk', 'Unknown').title()}")
    output.append(f"   Anxiety Risk: {mh.get('anxiety_risk', 'Unknown').title()}")
    output.append(f"   Depression Risk: {mh.get('depression_risk', 'Unknown').title()}")
    output.append("")
    
    # Confidence Scores
    confidence_scores = mh.get('confidence_scores', {})
    if confidence_scores:
        output.append("🎯 Confidence Scores:")
        for metric, score in confidence_scores.items():
            output.append(f"   {metric.title()}: {score:.0%}")
        output.append("")
    
    # Emotion Analysis
    if emotion:
        output.append("😊 Emotion Analysis:")
        output.append(f"   Primary Emotion: {emotion.get('primary_emotion', 'Unknown').title()}")
        output.append(f"   Confidence: {emotion.get('confidence', 0):.0%}")
        output.append("")
    
    # Sentiment Analysis
    if sentiment:
        output.append("💭 Sentiment Analysis:")
        output.append(f"   Sentiment: {sentiment.get('sentiment', 'Unknown').title()}")
        output.append(f"   Polarity: {sentiment.get('polarity_score', 0):.2f}")
        output.append(f"   Confidence: {sentiment.get('confidence', 0):.0%}")
        output.append("")
    
    # Crisis Response
    if crisis.get('intervention_needed'):
        output.append("🚨 Crisis Alert:")
        output.append("   Immediate professional help recommended!")
        output.append("   National Suicide Prevention Lifeline: 988")
        output.append("   Crisis Text Line: Text HOME to 741741")
        output.append("")
    
    # Recommendations
    recommendations = mh.get('recommendations', [])
    if recommendations:
        output.append("💡 Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            output.append(f"   {i}. {rec}")
        output.append("")
    
    output.append("⚠️  Remember: This analysis is not a medical diagnosis.")
    output.append("   Please consult mental health professionals for concerns.")
    
    return "\n".join(output)


def format_results_summary(results: dict) -> str:
    """Format results as a brief summary."""
    if results.get('error'):
        return f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}"
    
    mh = results.get('mental_health_analysis', {})
    emotion = results.get('emotion_analysis', {})
    sentiment = results.get('sentiment_analysis', {})
    crisis = results.get('crisis_response', {})
    
    overall_risk = mh.get('overall_risk', 'unknown')
    overall_conf = mh.get('confidence_scores', {}).get('overall', 0)
    primary_emotion = emotion.get('primary_emotion', 'unknown')
    sentiment_label = sentiment.get('sentiment', 'unknown')
    
    # Risk level emoji
    risk_emoji = {
        'low': '🟢',
        'medium': '🟡', 
        'high': '🔴',
        'crisis': '🚨'
    }.get(overall_risk.lower(), '⚪')
    
    summary = f"{risk_emoji} Risk: {overall_risk.title()} ({overall_conf:.0%}) | "
    summary += f"😊 Emotion: {primary_emotion.title()} | "
    summary += f"💭 Sentiment: {sentiment_label.title()}"
    
    if crisis.get('intervention_needed'):
        summary += "\n🚨 CRISIS ALERT: Seek immediate professional help! Call 988"
    
    return summary


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load text
    if args.text:
        text = args.text
    else:
        text = load_text_from_file(args.file)
    
    if not text.strip():
        print("❌ Error: No text provided for analysis.", file=sys.stderr)
        sys.exit(1)
    
    # Initialize analysis engine
    if not args.quiet:
        print("🔄 Initializing Mental Health Detector...", file=sys.stderr)
    
    try:
        engine = AnalysisEngine()
        engine.initialize()
    except Exception as e:
        print(f"❌ Error initializing analysis engine: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Perform analysis
    if not args.quiet:
        print("🧠 Analyzing text...", file=sys.stderr)
    
    try:
        results = engine.analyze_text(text)
    except Exception as e:
        print(f"❌ Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Format output
    if args.format == "json":
        output = format_results_json(results)
    elif args.format == "table":
        output = format_results_table(results)
    else:  # summary
        output = format_results_summary(results)
    
    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            if not args.quiet:
                print(f"✅ Results saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error writing to file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output)


if __name__ == "__main__":
    main()
