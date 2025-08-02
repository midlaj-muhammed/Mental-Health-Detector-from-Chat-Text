"""
Mental Health Detector - Streamlit Web Application
A responsible AI application for analyzing text to identify potential mental health concerns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import sys
import os
from pathlib import Path

# Add src directory to path for local and cloud deployment
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

# Import our modules with multiple fallback strategies
try:
    # Try direct import first (for Streamlit Cloud)
    from src.utils.analysis_engine import AnalysisEngine
    from src.utils.ethical_ai import EthicalAI
    from src.models.mental_health_classifier import MentalHealthClassifier
except ImportError:
    try:
        # Try relative import (for local development)
        from utils.analysis_engine import AnalysisEngine
        from utils.ethical_ai import EthicalAI
        from models.mental_health_classifier import MentalHealthClassifier
    except ImportError as e:
        st.error(f"Error importing modules: {e}")
        st.error("Please ensure all required modules are installed and the project structure is correct.")
        st.info("Try running: `pip install -r requirements.txt` and `python setup_models.py`")
        st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        font-family: 'Times New Roman', serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        border-bottom: 3px solid #34495e;
        padding-bottom: 1rem;
    }
    .disclaimer-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff9800;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #e65100 !important;
    }
    .disclaimer-box h3 {
        color: #bf360c !important;
        font-weight: bold;
    }
    .disclaimer-box p, .disclaimer-box li {
        color: #e65100 !important;
        font-weight: 500;
    }
    .disclaimer-box strong {
        color: #bf360c !important;
    }
    .crisis-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border: 2px solid #e74c3c;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        50% { box-shadow: 0 8px 12px rgba(231,76,60,0.4); }
        100% { box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .risk-low { border-left-color: #27ae60 !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-high { border-left-color: #e74c3c !important; }
    .risk-crisis { border-left-color: #8e44ad !important; }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 8px;
        transition: width 0.5s ease;
    }
    .confidence-low { background: linear-gradient(90deg, #e74c3c, #c0392b); }
    .confidence-medium { background: linear-gradient(90deg, #f39c12, #e67e22); }
    .confidence-high { background: linear-gradient(90deg, #27ae60, #229954); }

    .emotion-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    .emotion-joy { background: #2ecc71; color: white; }
    .emotion-sadness { background: #3498db; color: white; }
    .emotion-anger { background: #e74c3c; color: white; }
    .emotion-fear { background: #9b59b6; color: white; }
    .emotion-surprise { background: #f39c12; color: white; }
    .emotion-disgust { background: #95a5a6; color: white; }
    .emotion-neutral { background: #bdc3c7; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_analysis_engine():
    """Initialize and cache the analysis engine"""
    try:
        engine = AnalysisEngine()
        engine.initialize()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize analysis engine: {e}")
        return None

def display_disclaimer():
    """Display important disclaimer with high contrast text"""
    st.markdown("""
    <div class="disclaimer-box">
        <h3 style="color: #bf360c !important; font-size: 1.3rem; margin-bottom: 1rem;">⚠️ Important Disclaimer</h3>
        <p style="color: #e65100 !important; font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">
            <strong style="color: #bf360c !important;">This application is NOT a substitute for professional mental health care.</strong>
        </p>
        <p style="color: #e65100 !important; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem;">
            This tool is designed as a supportive resource to help identify potential concerns that should be discussed with qualified mental health professionals. If you or someone you know is in crisis, please contact emergency services or a crisis hotline immediately.
        </p>
        <ul style="color: #e65100 !important; font-size: 0.95rem; line-height: 1.6;">
            <li style="margin-bottom: 0.5rem;"><strong style="color: #bf360c !important;">Emergency:</strong> 911</li>
            <li style="margin-bottom: 0.5rem;"><strong style="color: #bf360c !important;">National Suicide Prevention Lifeline:</strong> 988</li>
            <li style="margin-bottom: 0.5rem;"><strong style="color: #bf360c !important;">Crisis Text Line:</strong> Text HOME to 741741</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_crisis_alert(crisis_response):
    """Display crisis intervention alert"""
    if crisis_response.get('intervention_needed', False):
        st.markdown("""
        <div class="crisis-alert">
            <h3>🚨 IMMEDIATE ATTENTION NEEDED</h3>
            <p><strong>The analysis indicates potential crisis indicators. Please take immediate action:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        for action in crisis_response.get('immediate_actions', []):
            st.error(f"• {action}")
        
        st.markdown("---")

def create_risk_gauge(risk_level, confidence):
    """Create an enhanced gauge chart for risk level"""
    # Map risk levels to numeric values and colors
    risk_mapping = {
        'low': {'value': 1, 'color': '#27ae60', 'bg': '#d5f4e6'},
        'medium': {'value': 2, 'color': '#f39c12', 'bg': '#fef9e7'},
        'high': {'value': 3, 'color': '#e74c3c', 'bg': '#fadbd8'},
        'crisis': {'value': 4, 'color': '#8e44ad', 'bg': '#f4ecf7'}
    }

    risk_info = risk_mapping.get(risk_level.lower(), risk_mapping['low'])
    risk_value = risk_info['value']

    # Create enhanced gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{risk_level.title()} Risk</b><br><span style='font-size:0.8em'>Confidence: {confidence:.0%}</span>",
            'font': {'size': 20}
        },
        number = {'font': {'size': 40, 'color': risk_info['color']}},
        gauge = {
            'axis': {
                'range': [0, 4],
                'tickwidth': 1,
                'tickcolor': "darkblue",
                'tickvals': [1, 2, 3, 4],
                'ticktext': ['Low', 'Medium', 'High', 'Crisis']
            },
            'bar': {'color': risk_info['color'], 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': '#d5f4e6'},
                {'range': [1, 2], 'color': '#fef9e7'},
                {'range': [2, 3], 'color': '#fadbd8'},
                {'range': [3, 4], 'color': '#f4ecf7'}
            ],
            'threshold': {
                'line': {'color': risk_info['color'], 'width': 4},
                'thickness': 0.8,
                'value': risk_value
            }
        }
    ))

    fig.update_layout(
        height=350,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor=risk_info['bg'],
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'low': '#27ae60',
        'medium': '#f39c12',
        'high': '#e74c3c',
        'crisis': '#8e44ad'
    }
    return colors.get(risk_level.lower(), '#95a5a6')

def get_confidence_class(confidence):
    """Get CSS class for confidence level"""
    if confidence >= 0.7:
        return 'confidence-high'
    elif confidence >= 0.4:
        return 'confidence-medium'
    else:
        return 'confidence-low'

def display_enhanced_metric_card(title, risk_level, confidence, icon="🎯"):
    """Display an enhanced metric card"""
    risk_color = get_risk_color(risk_level)
    confidence_class = get_confidence_class(confidence)

    st.markdown(f"""
    <div class="metric-card risk-{risk_level.lower()}">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 2rem; margin-right: 10px;">{icon}</span>
            <h3 style="margin: 0; color: #2c3e50;">{title}</h3>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: {risk_color}; margin-bottom: 10px;">
            {risk_level.title()}
        </div>
        <div style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 8px;">
            Confidence: {confidence:.0%}
        </div>
        <div class="{confidence_class} confidence-bar" style="width: {confidence*100}%;"></div>
    </div>
    """, unsafe_allow_html=True)

def display_analysis_results(results):
    """Display comprehensive analysis results with enhanced UI"""
    if not results or results.get('error'):
        st.error(f"❌ Analysis error: {results.get('error_message', 'Unknown error')}")
        return

    # Check for crisis response first
    crisis_response = results.get('crisis_response', {})
    display_crisis_alert(crisis_response)

    # Main analysis results
    mental_health = results.get('mental_health_analysis', {})

    st.markdown("## 📊 Analysis Results")

    # Enhanced metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        display_enhanced_metric_card(
            "Overall Risk",
            mental_health.get('overall_risk', 'Unknown'),
            mental_health.get('confidence_scores', {}).get('overall', 0),
            "🎯"
        )

    with col2:
        display_enhanced_metric_card(
            "Anxiety Risk",
            mental_health.get('anxiety_risk', 'Unknown'),
            mental_health.get('confidence_scores', {}).get('anxiety', 0),
            "😰"
        )

    with col3:
        display_enhanced_metric_card(
            "Depression Risk",
            mental_health.get('depression_risk', 'Unknown'),
            mental_health.get('confidence_scores', {}).get('depression', 0),
            "😢"
        )
    
    # Risk gauge and detailed visualization
    st.markdown("---")
    st.markdown("## 📈 Risk Assessment Visualization")

    overall_risk = mental_health.get('overall_risk', 'low')
    overall_confidence = mental_health.get('confidence_scores', {}).get('overall', 0)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = create_risk_gauge(overall_risk, overall_confidence)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Confidence Breakdown")
        confidence_scores = mental_health.get('confidence_scores', {})

        for metric, score in confidence_scores.items():
            confidence_class = get_confidence_class(score)
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold;">{metric.title()}</span>
                    <span style="color: #7f8c8d;">{score:.0%}</span>
                </div>
                <div class="{confidence_class} confidence-bar" style="width: {score*100}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced emotion analysis
    st.markdown("---")
    st.markdown("## 😊 Emotion & Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        emotion_analysis = results.get('emotion_analysis', {})
        if emotion_analysis:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🎭 Emotion Detection")

            primary_emotion = emotion_analysis.get('primary_emotion', 'Unknown')
            emotion_confidence = emotion_analysis.get('confidence', 0)

            # Display primary emotion with badge
            emotion_class = f"emotion-{primary_emotion.lower()}"
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <div class="emotion-badge {emotion_class}" style="font-size: 1.2rem; padding: 10px 20px;">
                    {primary_emotion.title()} ({emotion_confidence:.0%})
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Emotion distribution chart
            all_emotions = emotion_analysis.get('all_emotions', {})
            if all_emotions:
                emotion_df = pd.DataFrame(list(all_emotions.items()), columns=['Emotion', 'Score'])
                emotion_df = emotion_df.sort_values('Score', ascending=True)

                fig = px.bar(
                    emotion_df,
                    x='Score',
                    y='Emotion',
                    orientation='h',
                    title='Emotion Distribution',
                    color='Score',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        sentiment_analysis = results.get('sentiment_analysis', {})
        if sentiment_analysis:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 💭 Sentiment Analysis")

            sentiment = sentiment_analysis.get('sentiment', 'Unknown')
            sentiment_confidence = sentiment_analysis.get('confidence', 0)
            polarity_score = sentiment_analysis.get('polarity_score', 0)
            severity_level = sentiment_analysis.get('severity_level', 'Unknown')

            # Sentiment visualization
            sentiment_color = get_risk_color('low' if sentiment == 'positive' else 'high' if sentiment == 'negative' else 'medium')

            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 1.5rem; font-weight: bold; color: {sentiment_color};">
                    {sentiment.title()}
                </div>
                <div style="font-size: 1rem; color: #7f8c8d; margin: 10px 0;">
                    Confidence: {sentiment_confidence:.0%}
                </div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">
                    Polarity: {polarity_score:.2f} | Severity: {severity_level.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Polarity gauge
            fig_polarity = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = polarity_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Polarity"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': sentiment_color},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "#fadbd8"},
                        {'range': [-0.3, 0.3], 'color': "#fef9e7"},
                        {'range': [0.3, 1], 'color': "#d5f4e6"}
                    ],
                    'threshold': {
                        'line': {'color': sentiment_color, 'width': 4},
                        'thickness': 0.75,
                        'value': polarity_score
                    }
                }
            ))
            fig_polarity.update_layout(height=250)
            st.plotly_chart(fig_polarity, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Risk Factors and Recommendations
    st.markdown("---")
    st.markdown("## ⚖️ Risk Assessment Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="result-card risk-high">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Risk Factors")
        risk_factors = mental_health.get('risk_factors', [])
        if risk_factors:
            for factor in risk_factors:
                # Clean up factor names
                clean_factor = factor.replace('_', ' ').title()
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <span style="color: #e74c3c; margin-right: 8px;">⚠️</span>
                    <span>{clean_factor}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; color: #27ae60; padding: 20px;">
                <span style="font-size: 2rem;">✅</span><br>
                <strong>No specific risk factors identified</strong>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card risk-low">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Protective Factors")
        protective_factors = mental_health.get('protective_factors', [])
        if protective_factors:
            for factor in protective_factors:
                clean_factor = factor.replace('_', ' ').title()
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <span style="color: #27ae60; margin-right: 8px;">🛡️</span>
                    <span>{clean_factor}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; color: #f39c12; padding: 20px;">
                <span style="font-size: 2rem;">💡</span><br>
                <strong>Consider building protective factors</strong>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Recommendations
    st.markdown("---")
    st.markdown("## 💡 Personalized Recommendations")

    recommendations = mental_health.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            if rec.startswith('IMMEDIATE:'):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #c0392b;">
                    <strong>🚨 IMMEDIATE ACTION #{i}</strong><br>
                    {rec.replace('IMMEDIATE:', '').strip()}
                </div>
                """, unsafe_allow_html=True)
            elif any(word in rec.lower() for word in ['strongly', 'urgent', 'crisis']):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f39c12, #e67e22); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #d68910;">
                    <strong>⚡ URGENT RECOMMENDATION #{i}</strong><br>
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #2471a3;">
                    <strong>💡 SUGGESTION #{i}</strong><br>
                    {rec}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations generated.")
    
    # Text features (expandable)
    with st.expander("Text Analysis Features"):
        text_features = results.get('text_features', {})
        if text_features:
            features_df = pd.DataFrame([text_features.get('features', {}).__dict__])
            st.dataframe(features_df)
    
    # Ethical considerations
    with st.expander("Ethical Considerations & Limitations"):
        ethical = results.get('ethical_considerations', {})
        st.write("**Disclaimer:**", ethical.get('disclaimer', ''))
        
        limitations = ethical.get('limitations', [])
        if limitations:
            st.write("**Limitations:**")
            for limitation in limitations:
                st.write(f"• {limitation}")
        
        bias_analysis = ethical.get('bias_analysis')
        if bias_analysis:
            st.write("**Bias Analysis:**")
            st.json(bias_analysis)

def main():
    """Main application function"""
    # Classic Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">Mental Health Detector</h1>
        <p style="font-size: 1.1rem; color: #5d6d7e; margin-top: 1rem; font-style: italic;">
            AI-Powered Mental Health Analysis with Ethical Safeguards
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 1.5rem;">
            <span style="background: #34495e; color: white; padding: 8px 16px; border-radius: 5px; font-size: 0.9rem; font-weight: bold;">
                🔒 Privacy Protected
            </span>
            <span style="background: #2c3e50; color: white; padding: 8px 16px; border-radius: 5px; font-size: 0.9rem; font-weight: bold;">
                🤖 AI-Powered
            </span>
            <span style="background: #c0392b; color: white; padding: 8px 16px; border-radius: 5px; font-size: 0.9rem; font-weight: bold;">
                ⚕️ Not Medical Advice
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display disclaimer
    display_disclaimer()
    
    # Initialize analysis engine
    analysis_engine = initialize_analysis_engine()
    if not analysis_engine:
        st.error("Failed to initialize the analysis system. Please try again later.")
        return
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    enable_bias_detection = st.sidebar.checkbox("Enable Bias Detection", value=True)
    show_confidence_scores = st.sidebar.checkbox("Show Confidence Scores", value=True)
    show_detailed_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)
    
    # Privacy settings
    st.sidebar.subheader("Privacy Settings")
    encrypt_data = st.sidebar.checkbox("Encrypt Data", value=True)
    st.sidebar.info("No data is stored permanently. All analysis is performed locally.")
    
    # Enhanced input area
    st.markdown("---")
    st.markdown("## 📝 Text Analysis")

    # Input method selection with better styling
    st.markdown("### Choose Input Method")
    col1, col2 = st.columns(2)

    with col1:
        type_text = st.button("📝 Type/Paste Text", use_container_width=True)
    with col2:
        upload_file = st.button("📁 Upload Text File", use_container_width=True)

    # Initialize session state for input method
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "type"

    if type_text:
        st.session_state.input_method = "type"
    elif upload_file:
        st.session_state.input_method = "upload"

    text_input = ""

    if st.session_state.input_method == "type":
        st.markdown("### ✍️ Enter Your Text")
        text_input = st.text_area(
            "Share your thoughts, feelings, or any text you'd like analyzed:",
            height=200,
            max_chars=5000,
            placeholder="Example: 'I've been feeling really anxious lately about work and can't seem to relax...'\n\nYour privacy is protected - no data is stored permanently.",
            help="💡 Tip: Be honest and detailed for better analysis. The more context you provide, the more accurate the assessment."
        )

        # Character counter
        if text_input:
            char_count = len(text_input)
            color = "#27ae60" if char_count >= 50 else "#f39c12" if char_count >= 20 else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: right; color: {color}; font-size: 0.9rem; margin-top: -10px;">
                {char_count}/5000 characters
                {" ✅ Good length for analysis" if char_count >= 50 else " ⚠️ Consider adding more detail" if char_count >= 20 else " ❌ Too short for meaningful analysis"}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("### 📁 Upload Text File")
        uploaded_file = st.file_uploader(
            "Choose a text file to analyze",
            type=['txt', 'md', 'doc'],
            help="📄 Supported formats: .txt, .md, .doc files"
        )

        if uploaded_file is not None:
            try:
                text_input = str(uploaded_file.read(), "utf-8")
                st.success(f"✅ File uploaded successfully! ({len(text_input)} characters)")

                with st.expander("📖 Preview uploaded text"):
                    st.text_area("File content:", value=text_input[:1000] + ("..." if len(text_input) > 1000 else ""), height=150, disabled=True)

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.info("💡 Please ensure your file is a valid text file in UTF-8 encoding.")
    
    # Enhanced analysis section
    st.markdown("---")

    # Analysis readiness check
    can_analyze = bool(text_input.strip() and len(text_input.strip()) >= 10)

    if can_analyze:
        st.markdown("### 🚀 Ready for Analysis")
        st.success("✅ Text is ready for mental health analysis")
    else:
        st.markdown("### ⏳ Preparing for Analysis")
        if not text_input.strip():
            st.warning("📝 Please enter some text to analyze")
        elif len(text_input.strip()) < 10:
            st.warning("📏 Please enter at least 10 characters for meaningful analysis")

    # Enhanced analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🧠 Analyze Mental Health",
            type="primary",
            disabled=not can_analyze,
            use_container_width=True,
            help="Click to start AI-powered mental health analysis"
        )

    if analyze_button:
        if not text_input.strip():
            st.error("❌ Please enter some text to analyze.")
            return

        # Enhanced progress display
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step-by-step progress
            status_text.text("🔄 Initializing analysis...")
            progress_bar.progress(10)

            status_text.text("🧠 Loading AI models...")
            progress_bar.progress(30)

            status_text.text("📝 Processing text...")
            progress_bar.progress(50)

            status_text.text("😊 Analyzing emotions...")
            progress_bar.progress(70)

            status_text.text("💭 Evaluating sentiment...")
            progress_bar.progress(85)

            # Perform analysis
            results = analysis_engine.analyze_text(
                text=text_input,
                user_id=None,  # Anonymous
                session_id=st.session_state.get('session_id')
            )

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.balloons()  # Celebration animation
            display_analysis_results(results)

            # Enhanced export options
            st.markdown("---")
            st.markdown("### 📊 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                results_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=results_json,
                    file_name=f"mental_health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                # Create a summary text
                mh = results.get('mental_health_analysis', {})
                summary_text = f"""Mental Health Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Risk: {mh.get('overall_risk', 'Unknown')}
Anxiety Risk: {mh.get('anxiety_risk', 'Unknown')}
Depression Risk: {mh.get('depression_risk', 'Unknown')}

Primary Emotion: {results.get('emotion_analysis', {}).get('primary_emotion', 'Unknown')}
Sentiment: {results.get('sentiment_analysis', {}).get('sentiment', 'Unknown')}

Recommendations:
{chr(10).join(f"• {rec}" for rec in mh.get('recommendations', [])[:5])}

Disclaimer: This analysis is not a medical diagnosis and should not replace professional mental health care.
"""
                st.download_button(
                    label="📝 Download Summary",
                    data=summary_text,
                    file_name=f"mental_health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col3:
                if st.button("🔄 Analyze Again", use_container_width=True):
                    st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Please try again or contact support if the issue persists.")
            logger.error(f"Analysis error: {e}")
    
    # Help section
    with st.expander("Help & Usage Guidelines"):
        st.markdown("""
        ### How to Use This Tool
        
        1. **Enter Text**: Type or paste text you want to analyze (journal entries, messages, etc.)
        2. **Click Analyze**: The system will process your text using AI models
        3. **Review Results**: Examine the risk assessment and recommendations
        4. **Seek Professional Help**: If indicated, contact a mental health professional
        
        ### What This Tool Does
        - Analyzes emotional content and sentiment
        - Identifies potential indicators of anxiety or depression
        - Provides risk assessment with confidence scores
        - Offers recommendations for next steps
        
        ### What This Tool Does NOT Do
        - Provide medical diagnosis
        - Replace professional mental health care
        - Store or share your personal data
        - Guarantee accuracy of assessments
        
        ### Privacy & Security
        - No data is stored permanently
        - All processing is done locally when possible
        - Optional encryption for sensitive data
        - Anonymous usage - no personal information required
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Mental Health Detector v1.0 | Built with ❤️ for mental health awareness</p>
        <p>Remember: This tool is not a substitute for professional mental health care.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    main()
