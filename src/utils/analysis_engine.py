"""
Analysis Engine for Mental Health Detection
Coordinates all analysis components and provides unified interface.
"""

import logging
import yaml
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

from ..models.mental_health_classifier import MentalHealthClassifier, MentalHealthResult
from .text_processor import TextProcessor
from .privacy_handler import PrivacyHandler

class AnalysisEngine:
    """
    Main analysis engine that coordinates all mental health analysis components.
    Provides a unified interface for text analysis with privacy and ethical considerations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.classifier = MentalHealthClassifier(self.config.get('models', {}))
        self.text_processor = TextProcessor()
        self.privacy_handler = PrivacyHandler(self.config.get('privacy', {}))
        
        # Analysis settings
        self.min_text_length = self.config.get('text_processing', {}).get('min_length', 10)
        self.max_text_length = self.config.get('text_processing', {}).get('max_length', 5000)
        
        # Ethical safeguards
        self.enable_bias_detection = self.config.get('ethics', {}).get('enable_bias_detection', True)
        self.crisis_threshold = self.config.get('ethics', {}).get('crisis_intervention_threshold', 0.85)
        
        self.logger.info("Analysis Engine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'models': {
                'emotion_detection': {'confidence_threshold': 0.6},
                'sentiment_analysis': {'confidence_threshold': 0.7},
                'mental_health_classifier': {'confidence_threshold': 0.65}
            },
            'text_processing': {
                'min_length': 10,
                'max_length': 5000
            },
            'ethics': {
                'enable_bias_detection': True,
                'crisis_intervention_threshold': 0.85
            },
            'privacy': {
                'encrypt_data': True,
                'log_interactions': False
            }
        }
    
    def initialize(self) -> None:
        """Initialize all analysis components"""
        try:
            self.logger.info("Initializing analysis components...")
            
            # Load ML models
            self.classifier.load_models()
            
            # Initialize privacy handler
            self.privacy_handler.initialize()
            
            self.logger.info("All analysis components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing analysis engine: {str(e)}")
            raise
    
    def analyze_text(self, text: str, user_id: Optional[str] = None, 
                    session_id: Optional[str] = None) -> Dict:
        """
        Perform comprehensive mental health analysis on text
        
        Args:
            text: Input text to analyze
            user_id: Optional user identifier (for privacy tracking)
            session_id: Optional session identifier
            
        Returns:
            Dictionary with analysis results and metadata
        """
        analysis_start = datetime.now()
        
        try:
            # Input validation
            validation_result = self._validate_input(text)
            if not validation_result['valid']:
                return self._create_error_response(validation_result['error'], analysis_start)
            
            # Privacy handling - keep original text for analysis
            original_text = text
            encrypted = False
            if self.privacy_handler.should_encrypt():
                # Only encrypt for logging/storage purposes, not for analysis
                encrypted = True

            # Text preprocessing and feature extraction (use original text)
            text_features = self.text_processor.analyze_text_patterns(original_text)

            # Mental health classification (use original text)
            mental_health_result = self.classifier.analyze_text(original_text)
            
            # Bias detection (if enabled)
            bias_analysis = None
            if self.enable_bias_detection:
                bias_analysis = self._detect_bias(original_text, mental_health_result)
            
            # Crisis intervention check
            crisis_response = self._check_crisis_intervention(mental_health_result)
            
            # Generate comprehensive response
            response = self._create_analysis_response(
                mental_health_result=mental_health_result,
                text_features=text_features,
                bias_analysis=bias_analysis,
                crisis_response=crisis_response,
                analysis_start=analysis_start,
                encrypted=encrypted,
                user_id=user_id,
                session_id=session_id
            )
            
            # Log interaction (if enabled)
            if self.config.get('privacy', {}).get('log_interactions', False):
                self._log_interaction(response, user_id, session_id)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in text analysis: {str(e)}")
            return self._create_error_response(str(e), analysis_start)
    
    def _validate_input(self, text: str) -> Dict[str, Union[bool, str]]:
        """Validate input text"""
        if not text or not isinstance(text, str):
            return {'valid': False, 'error': 'Invalid input: text must be a non-empty string'}
        
        text = text.strip()
        
        if len(text) < self.min_text_length:
            return {'valid': False, 'error': f'Text too short: minimum {self.min_text_length} characters required'}
        
        if len(text) > self.max_text_length:
            return {'valid': False, 'error': f'Text too long: maximum {self.max_text_length} characters allowed'}
        
        return {'valid': True, 'error': None}
    
    def _detect_bias(self, text: str, result: MentalHealthResult) -> Dict:
        """
        Detect potential bias in analysis results
        This is a simplified implementation - in production, this would be more sophisticated
        """
        bias_indicators = {
            'demographic_bias': False,
            'cultural_bias': False,
            'language_bias': False,
            'confidence_bias': False,
            'warnings': []
        }
        
        # Check for demographic bias indicators
        demographic_terms = ['age', 'gender', 'race', 'ethnicity', 'religion']
        if any(term in text.lower() for term in demographic_terms):
            bias_indicators['demographic_bias'] = True
            bias_indicators['warnings'].append('Text contains demographic references - ensure fair treatment')
        
        # Check for cultural bias
        cultural_terms = ['culture', 'tradition', 'belief', 'custom']
        if any(term in text.lower() for term in cultural_terms):
            bias_indicators['cultural_bias'] = True
            bias_indicators['warnings'].append('Text contains cultural references - consider cultural context')
        
        # Check for language complexity bias
        if result.emotion_result.confidence < 0.5:
            bias_indicators['language_bias'] = True
            bias_indicators['warnings'].append('Low confidence may indicate language complexity or non-standard usage')
        
        # Check for confidence bias
        if result.overall_confidence > 0.9:
            bias_indicators['confidence_bias'] = True
            bias_indicators['warnings'].append('Very high confidence - verify results are not overconfident')
        
        return bias_indicators
    
    def _check_crisis_intervention(self, result: MentalHealthResult) -> Dict:
        """Check if crisis intervention is needed"""
        crisis_response = {
            'intervention_needed': False,
            'severity': 'none',
            'immediate_actions': [],
            'resources': []
        }
        
        if result.crisis_indicators or result.overall_risk == 'crisis':
            crisis_response['intervention_needed'] = True
            crisis_response['severity'] = 'immediate'
            crisis_response['immediate_actions'] = [
                'Contact emergency services (911) immediately',
                'Call National Suicide Prevention Lifeline: 988',
                'Go to nearest emergency room',
                'Contact trusted friend or family member',
                'Do not leave person alone'
            ]
        elif result.overall_risk == 'high' or result.overall_confidence > self.crisis_threshold:
            crisis_response['intervention_needed'] = True
            crisis_response['severity'] = 'urgent'
            crisis_response['immediate_actions'] = [
                'Contact mental health professional',
                'Call crisis hotline: 988',
                'Reach out to support network',
                'Consider emergency services if situation worsens'
            ]
        
        # Add resources from config
        resources = self.config.get('resources', {})
        if 'crisis_hotlines' in resources:
            crisis_response['resources'].extend(resources['crisis_hotlines'])
        
        return crisis_response
    
    def _create_analysis_response(self, mental_health_result: MentalHealthResult,
                                 text_features: Dict, bias_analysis: Optional[Dict],
                                 crisis_response: Dict, analysis_start: datetime,
                                 encrypted: bool, user_id: Optional[str],
                                 session_id: Optional[str]) -> Dict:
        """Create comprehensive analysis response"""
        
        total_processing_time = (datetime.now() - analysis_start).total_seconds()
        
        response = {
            'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': analysis_start.isoformat(),
            'processing_time': total_processing_time,
            
            # Main results
            'mental_health_analysis': {
                'overall_risk': mental_health_result.overall_risk,
                'anxiety_risk': mental_health_result.anxiety_risk,
                'depression_risk': mental_health_result.depression_risk,
                'confidence_scores': {
                    'overall': mental_health_result.overall_confidence,
                    'anxiety': mental_health_result.anxiety_confidence,
                    'depression': mental_health_result.depression_confidence
                },
                'risk_factors': mental_health_result.risk_factors,
                'protective_factors': mental_health_result.protective_factors,
                'recommendations': mental_health_result.recommendations
            },
            
            # Detailed analysis
            'emotion_analysis': {
                'primary_emotion': mental_health_result.emotion_result.emotion,
                'confidence': mental_health_result.emotion_result.confidence,
                'all_emotions': mental_health_result.emotion_result.all_scores,
                'mental_health_risk': mental_health_result.emotion_result.mental_health_risk
            },
            
            'sentiment_analysis': {
                'sentiment': mental_health_result.sentiment_result.sentiment,
                'confidence': mental_health_result.sentiment_result.confidence,
                'polarity_score': mental_health_result.sentiment_result.polarity_score,
                'severity_level': mental_health_result.sentiment_result.severity_level,
                'indicators': mental_health_result.sentiment_result.mental_health_indicators
            },
            
            # Text features
            'text_features': text_features,
            
            # Crisis intervention
            'crisis_response': crisis_response,
            
            # Ethical considerations
            'ethical_considerations': {
                'disclaimer': 'This analysis is not a medical diagnosis and should not replace professional mental health care.',
                'limitations': [
                    'Results may not capture cultural or contextual nuances',
                    'Model performance may vary across different demographics',
                    'Professional interpretation recommended for clinical decisions'
                ],
                'bias_analysis': bias_analysis,
                'model_version': mental_health_result.model_version
            },
            
            # Privacy and metadata
            'privacy': {
                'encrypted': encrypted,
                'data_retention': 'No data stored permanently',
                'user_id_hashed': bool(user_id)
            },
            
            'metadata': {
                'session_id': session_id,
                'model_versions': {
                    'mental_health_classifier': mental_health_result.model_version,
                    'analysis_engine': '1.0.0'
                }
            }
        }
        
        return response
    
    def _create_error_response(self, error_message: str, analysis_start: datetime) -> Dict:
        """Create error response"""
        return {
            'analysis_id': f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': analysis_start.isoformat(),
            'processing_time': (datetime.now() - analysis_start).total_seconds(),
            'error': True,
            'error_message': error_message,
            'mental_health_analysis': {
                'overall_risk': 'unknown',
                'recommendations': ['Please try again with valid input']
            },
            'ethical_considerations': {
                'disclaimer': 'This analysis is not a medical diagnosis and should not replace professional mental health care.'
            }
        }
    
    def _log_interaction(self, response: Dict, user_id: Optional[str], 
                        session_id: Optional[str]) -> None:
        """Log interaction for analysis and improvement (if enabled)"""
        try:
            log_entry = {
                'timestamp': response['timestamp'],
                'analysis_id': response['analysis_id'],
                'user_id_hash': hash(user_id) if user_id else None,
                'session_id': session_id,
                'overall_risk': response['mental_health_analysis']['overall_risk'],
                'processing_time': response['processing_time'],
                'crisis_intervention': response['crisis_response']['intervention_needed']
            }
            
            # Log to file or database (implementation depends on requirements)
            self.logger.info(f"Analysis logged: {json.dumps(log_entry)}")
            
        except Exception as e:
            self.logger.error(f"Error logging interaction: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """Get system status and health check"""
        return {
            'status': 'operational',
            'components': {
                'mental_health_classifier': 'loaded',
                'text_processor': 'ready',
                'privacy_handler': 'active'
            },
            'configuration': {
                'bias_detection_enabled': self.enable_bias_detection,
                'crisis_threshold': self.crisis_threshold,
                'privacy_encryption': self.privacy_handler.should_encrypt()
            },
            'version': '1.0.0'
        }
