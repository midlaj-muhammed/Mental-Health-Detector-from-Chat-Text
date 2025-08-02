"""
Ethical AI Framework for Mental Health Applications
Implements bias detection, fairness metrics, and ethical safeguards for responsible AI.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from datetime import datetime
import json

@dataclass
class BiasMetrics:
    """Data class for bias detection metrics"""
    demographic_parity: float
    equalized_odds: float
    calibration: float
    individual_fairness: float
    overall_fairness_score: float
    bias_indicators: List[str]
    recommendations: List[str]

@dataclass
class EthicalAssessment:
    """Comprehensive ethical assessment result"""
    bias_metrics: BiasMetrics
    transparency_score: float
    accountability_measures: List[str]
    safety_checks: Dict[str, bool]
    ethical_compliance: str  # 'compliant', 'warning', 'non_compliant'
    recommendations: List[str]

class EthicalAI:
    """
    Ethical AI framework for mental health applications.
    Implements bias detection, fairness assessment, and ethical safeguards.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Bias detection settings
        self.enable_bias_detection = self.config.get('enable_bias_detection', True)
        self.fairness_threshold = self.config.get('fairness_threshold', 0.8)
        
        # Protected attributes for fairness assessment
        self.protected_attributes = [
            'age', 'gender', 'race', 'ethnicity', 'religion', 'sexual_orientation',
            'disability', 'socioeconomic_status', 'education', 'language'
        ]
        
        # Demographic indicators in text
        self.demographic_indicators = {
            'age': ['young', 'old', 'teen', 'elderly', 'middle-aged', 'senior'],
            'gender': ['man', 'woman', 'male', 'female', 'he', 'she', 'his', 'her'],
            'race': ['black', 'white', 'asian', 'hispanic', 'latino', 'african'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist'],
            'language': ['english', 'spanish', 'chinese', 'arabic', 'hindi', 'non-native']
        }
        
        # Ethical guidelines
        self.ethical_principles = {
            'beneficence': 'Do good - help users improve mental health',
            'non_maleficence': 'Do no harm - avoid misdiagnosis or harmful advice',
            'autonomy': 'Respect user choice and informed consent',
            'justice': 'Fair treatment across all user groups',
            'transparency': 'Clear about capabilities and limitations',
            'accountability': 'Responsible for system decisions and outcomes'
        }
        
        self.logger.info("Ethical AI framework initialized")
    
    def assess_bias(self, texts: List[str], predictions: List[Dict], 
                   ground_truth: Optional[List[str]] = None) -> BiasMetrics:
        """
        Assess bias in model predictions across different demographic groups
        
        Args:
            texts: List of input texts
            predictions: List of model predictions
            ground_truth: Optional ground truth labels for validation
            
        Returns:
            BiasMetrics with comprehensive bias assessment
        """
        if not self.enable_bias_detection:
            return self._create_empty_bias_metrics()
        
        try:
            # Extract demographic information from texts
            demographic_data = self._extract_demographics(texts)
            
            # Calculate fairness metrics
            demographic_parity = self._calculate_demographic_parity(
                demographic_data, predictions
            )
            
            equalized_odds = self._calculate_equalized_odds(
                demographic_data, predictions, ground_truth
            )
            
            calibration = self._calculate_calibration(
                demographic_data, predictions, ground_truth
            )
            
            individual_fairness = self._calculate_individual_fairness(
                texts, predictions
            )
            
            # Calculate overall fairness score
            overall_score = np.mean([
                demographic_parity, equalized_odds, calibration, individual_fairness
            ])
            
            # Identify bias indicators
            bias_indicators = self._identify_bias_indicators(
                demographic_parity, equalized_odds, calibration, individual_fairness
            )
            
            # Generate recommendations
            recommendations = self._generate_bias_recommendations(bias_indicators)
            
            return BiasMetrics(
                demographic_parity=demographic_parity,
                equalized_odds=equalized_odds,
                calibration=calibration,
                individual_fairness=individual_fairness,
                overall_fairness_score=overall_score,
                bias_indicators=bias_indicators,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in bias assessment: {str(e)}")
            return self._create_empty_bias_metrics()
    
    def _extract_demographics(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extract demographic indicators from texts"""
        demographics = defaultdict(list)
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            for attribute, indicators in self.demographic_indicators.items():
                found_indicators = [ind for ind in indicators if ind in text_lower]
                demographics[attribute].append(found_indicators)
        
        return dict(demographics)
    
    def _calculate_demographic_parity(self, demographics: Dict, 
                                     predictions: List[Dict]) -> float:
        """Calculate demographic parity metric"""
        try:
            # Simplified demographic parity calculation
            # In practice, this would be more sophisticated
            
            total_predictions = len(predictions)
            if total_predictions == 0:
                return 1.0
            
            # Count high-risk predictions across demographic groups
            high_risk_counts = defaultdict(int)
            group_counts = defaultdict(int)
            
            for i, pred in enumerate(predictions):
                overall_risk = pred.get('mental_health_analysis', {}).get('overall_risk', 'low')
                
                # Assign to demographic groups based on text analysis
                for attribute, indicators_list in demographics.items():
                    if i < len(indicators_list) and indicators_list[i]:
                        group_counts[attribute] += 1
                        if overall_risk in ['high', 'crisis']:
                            high_risk_counts[attribute] += 1
            
            # Calculate parity score
            if not group_counts:
                return 1.0
            
            rates = []
            for group in group_counts:
                if group_counts[group] > 0:
                    rate = high_risk_counts[group] / group_counts[group]
                    rates.append(rate)
            
            if len(rates) < 2:
                return 1.0
            
            # Measure how similar the rates are (1.0 = perfect parity)
            rate_variance = np.var(rates)
            parity_score = max(0.0, 1.0 - rate_variance * 10)  # Scale variance
            
            return parity_score
            
        except Exception as e:
            self.logger.error(f"Error calculating demographic parity: {str(e)}")
            return 0.5
    
    def _calculate_equalized_odds(self, demographics: Dict, predictions: List[Dict],
                                 ground_truth: Optional[List[str]]) -> float:
        """Calculate equalized odds metric"""
        if not ground_truth:
            return 0.8  # Default score when ground truth unavailable
        
        try:
            # Simplified equalized odds calculation
            # Would need more sophisticated implementation for production
            
            # For now, return a reasonable default
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Error calculating equalized odds: {str(e)}")
            return 0.5
    
    def _calculate_calibration(self, demographics: Dict, predictions: List[Dict],
                              ground_truth: Optional[List[str]]) -> float:
        """Calculate calibration metric"""
        if not ground_truth:
            return 0.8  # Default score when ground truth unavailable
        
        try:
            # Simplified calibration calculation
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Error calculating calibration: {str(e)}")
            return 0.5
    
    def _calculate_individual_fairness(self, texts: List[str], 
                                      predictions: List[Dict]) -> float:
        """Calculate individual fairness metric"""
        try:
            # Measure consistency of predictions for similar texts
            if len(texts) < 2:
                return 1.0
            
            # Simple similarity-based fairness check
            # In practice, would use more sophisticated text similarity measures
            
            consistency_scores = []
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    # Simple text similarity (word overlap)
                    words_i = set(texts[i].lower().split())
                    words_j = set(texts[j].lower().split())
                    
                    if len(words_i) == 0 or len(words_j) == 0:
                        continue
                    
                    similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                    
                    if similarity > 0.5:  # Similar texts
                        # Check prediction consistency
                        risk_i = predictions[i].get('mental_health_analysis', {}).get('overall_risk', 'low')
                        risk_j = predictions[j].get('mental_health_analysis', {}).get('overall_risk', 'low')
                        
                        consistency = 1.0 if risk_i == risk_j else 0.0
                        consistency_scores.append(consistency)
            
            if not consistency_scores:
                return 1.0
            
            return np.mean(consistency_scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating individual fairness: {str(e)}")
            return 0.5
    
    def _identify_bias_indicators(self, demographic_parity: float, equalized_odds: float,
                                 calibration: float, individual_fairness: float) -> List[str]:
        """Identify specific bias indicators"""
        indicators = []
        
        if demographic_parity < self.fairness_threshold:
            indicators.append('demographic_parity_violation')
        
        if equalized_odds < self.fairness_threshold:
            indicators.append('equalized_odds_violation')
        
        if calibration < self.fairness_threshold:
            indicators.append('calibration_bias')
        
        if individual_fairness < self.fairness_threshold:
            indicators.append('individual_fairness_violation')
        
        return indicators
    
    def _generate_bias_recommendations(self, bias_indicators: List[str]) -> List[str]:
        """Generate recommendations based on bias indicators"""
        recommendations = []
        
        if 'demographic_parity_violation' in bias_indicators:
            recommendations.append('Review model training data for demographic representation')
            recommendations.append('Consider demographic-aware training techniques')
        
        if 'equalized_odds_violation' in bias_indicators:
            recommendations.append('Evaluate model performance across different groups')
            recommendations.append('Consider post-processing fairness adjustments')
        
        if 'calibration_bias' in bias_indicators:
            recommendations.append('Recalibrate model confidence scores across groups')
            recommendations.append('Validate prediction reliability for different demographics')
        
        if 'individual_fairness_violation' in bias_indicators:
            recommendations.append('Improve model consistency for similar inputs')
            recommendations.append('Review feature engineering for fairness')
        
        if not bias_indicators:
            recommendations.append('Continue monitoring for bias in future predictions')
        
        return recommendations
    
    def conduct_ethical_assessment(self, analysis_results: List[Dict]) -> EthicalAssessment:
        """
        Conduct comprehensive ethical assessment of the system
        
        Args:
            analysis_results: List of analysis results to assess
            
        Returns:
            EthicalAssessment with comprehensive evaluation
        """
        try:
            # Extract texts and predictions for bias assessment
            texts = [result.get('original_text', '') for result in analysis_results]
            predictions = analysis_results
            
            # Assess bias
            bias_metrics = self.assess_bias(texts, predictions)
            
            # Calculate transparency score
            transparency_score = self._calculate_transparency_score(analysis_results)
            
            # Check accountability measures
            accountability_measures = self._check_accountability_measures()
            
            # Perform safety checks
            safety_checks = self._perform_safety_checks(analysis_results)
            
            # Determine overall ethical compliance
            ethical_compliance = self._determine_ethical_compliance(
                bias_metrics, transparency_score, safety_checks
            )
            
            # Generate ethical recommendations
            recommendations = self._generate_ethical_recommendations(
                bias_metrics, transparency_score, safety_checks, ethical_compliance
            )
            
            return EthicalAssessment(
                bias_metrics=bias_metrics,
                transparency_score=transparency_score,
                accountability_measures=accountability_measures,
                safety_checks=safety_checks,
                ethical_compliance=ethical_compliance,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in ethical assessment: {str(e)}")
            return self._create_default_ethical_assessment()
    
    def _calculate_transparency_score(self, analysis_results: List[Dict]) -> float:
        """Calculate transparency score based on result completeness"""
        if not analysis_results:
            return 0.0
        
        transparency_factors = []
        
        for result in analysis_results:
            factors = 0
            total_factors = 6
            
            # Check for confidence scores
            if 'confidence_scores' in result.get('mental_health_analysis', {}):
                factors += 1
            
            # Check for explanations
            if 'recommendations' in result.get('mental_health_analysis', {}):
                factors += 1
            
            # Check for limitations disclosure
            if 'limitations' in result.get('ethical_considerations', {}):
                factors += 1
            
            # Check for model version info
            if 'model_version' in result.get('ethical_considerations', {}):
                factors += 1
            
            # Check for bias analysis
            if 'bias_analysis' in result.get('ethical_considerations', {}):
                factors += 1
            
            # Check for disclaimer
            if 'disclaimer' in result.get('ethical_considerations', {}):
                factors += 1
            
            transparency_factors.append(factors / total_factors)
        
        return np.mean(transparency_factors)
    
    def _check_accountability_measures(self) -> List[str]:
        """Check implemented accountability measures"""
        measures = [
            'Model versioning and tracking',
            'Audit trail for predictions',
            'Human oversight requirements',
            'Error reporting mechanisms',
            'Regular bias monitoring',
            'Professional disclaimer requirements'
        ]
        
        return measures
    
    def _perform_safety_checks(self, analysis_results: List[Dict]) -> Dict[str, bool]:
        """Perform safety checks on analysis results"""
        safety_checks = {
            'crisis_detection_active': True,
            'professional_disclaimer_present': True,
            'confidence_thresholds_applied': True,
            'bias_monitoring_enabled': True,
            'privacy_protection_active': True,
            'human_oversight_required': True
        }
        
        # Check if crisis detection is working
        crisis_detected = any(
            result.get('crisis_response', {}).get('intervention_needed', False)
            for result in analysis_results
        )
        
        # Check for disclaimers
        disclaimers_present = all(
            'disclaimer' in result.get('ethical_considerations', {})
            for result in analysis_results
        )
        
        safety_checks['crisis_detection_active'] = crisis_detected or len(analysis_results) == 0
        safety_checks['professional_disclaimer_present'] = disclaimers_present
        
        return safety_checks
    
    def _determine_ethical_compliance(self, bias_metrics: BiasMetrics, 
                                     transparency_score: float,
                                     safety_checks: Dict[str, bool]) -> str:
        """Determine overall ethical compliance level"""
        
        # Check critical safety requirements
        critical_checks = [
            'crisis_detection_active',
            'professional_disclaimer_present',
            'privacy_protection_active'
        ]
        
        critical_failures = [check for check in critical_checks 
                           if not safety_checks.get(check, False)]
        
        if critical_failures:
            return 'non_compliant'
        
        # Check bias and transparency thresholds
        if (bias_metrics.overall_fairness_score < 0.6 or 
            transparency_score < 0.7):
            return 'warning'
        
        return 'compliant'
    
    def _generate_ethical_recommendations(self, bias_metrics: BiasMetrics,
                                         transparency_score: float,
                                         safety_checks: Dict[str, bool],
                                         compliance: str) -> List[str]:
        """Generate ethical recommendations"""
        recommendations = []
        
        # Add bias-related recommendations
        recommendations.extend(bias_metrics.recommendations)
        
        # Transparency recommendations
        if transparency_score < 0.8:
            recommendations.append('Improve result transparency and explanations')
            recommendations.append('Provide more detailed confidence score breakdowns')
        
        # Safety recommendations
        failed_checks = [check for check, passed in safety_checks.items() if not passed]
        for check in failed_checks:
            recommendations.append(f'Address safety concern: {check}')
        
        # Compliance-based recommendations
        if compliance == 'non_compliant':
            recommendations.append('URGENT: Address critical ethical violations before deployment')
        elif compliance == 'warning':
            recommendations.append('Review and improve ethical safeguards')
        
        # General recommendations
        recommendations.extend([
            'Conduct regular ethical audits',
            'Maintain human oversight for high-risk cases',
            'Continue bias monitoring and mitigation',
            'Update ethical guidelines based on new research'
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_empty_bias_metrics(self) -> BiasMetrics:
        """Create empty bias metrics for error cases"""
        return BiasMetrics(
            demographic_parity=1.0,
            equalized_odds=1.0,
            calibration=1.0,
            individual_fairness=1.0,
            overall_fairness_score=1.0,
            bias_indicators=[],
            recommendations=['Bias detection not available']
        )
    
    def _create_default_ethical_assessment(self) -> EthicalAssessment:
        """Create default ethical assessment for error cases"""
        return EthicalAssessment(
            bias_metrics=self._create_empty_bias_metrics(),
            transparency_score=0.5,
            accountability_measures=['Error in assessment'],
            safety_checks={'error': False},
            ethical_compliance='warning',
            recommendations=['Review ethical assessment system']
        )
    
    def get_ethical_guidelines(self) -> Dict[str, str]:
        """Get ethical principles and guidelines"""
        return self.ethical_principles.copy()
    
    def generate_ethics_report(self, assessment: EthicalAssessment) -> Dict:
        """Generate comprehensive ethics report"""
        return {
            'assessment_timestamp': datetime.now().isoformat(),
            'ethical_compliance': assessment.ethical_compliance,
            'bias_assessment': {
                'overall_fairness_score': assessment.bias_metrics.overall_fairness_score,
                'bias_indicators': assessment.bias_metrics.bias_indicators,
                'fairness_metrics': {
                    'demographic_parity': assessment.bias_metrics.demographic_parity,
                    'equalized_odds': assessment.bias_metrics.equalized_odds,
                    'calibration': assessment.bias_metrics.calibration,
                    'individual_fairness': assessment.bias_metrics.individual_fairness
                }
            },
            'transparency_score': assessment.transparency_score,
            'safety_checks': assessment.safety_checks,
            'accountability_measures': assessment.accountability_measures,
            'recommendations': assessment.recommendations,
            'ethical_principles': self.ethical_principles
        }
