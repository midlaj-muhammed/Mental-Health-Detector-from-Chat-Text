"""
Mental Health Resources Module
Provides comprehensive mental health resources, crisis intervention, and professional care recommendations.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import requests
from pathlib import Path

@dataclass
class CrisisResource:
    """Crisis intervention resource"""
    name: str
    phone: str
    text: Optional[str]
    website: str
    description: str
    availability: str  # "24/7", "business hours", etc.
    languages: List[str]
    specialties: List[str]  # "suicide", "domestic violence", etc.

@dataclass
class ProfessionalResource:
    """Professional mental health resource"""
    name: str
    type: str  # "therapist", "psychiatrist", "counselor", etc.
    website: str
    description: str
    search_features: List[str]
    cost_info: str
    insurance_accepted: bool

@dataclass
class SelfHelpResource:
    """Self-help and educational resource"""
    name: str
    type: str  # "app", "website", "book", "course", etc.
    url: Optional[str]
    description: str
    cost: str  # "free", "paid", "freemium"
    rating: Optional[float]
    categories: List[str]  # "anxiety", "depression", "mindfulness", etc.

class MentalHealthResourceProvider:
    """
    Comprehensive mental health resource provider.
    Manages crisis resources, professional care options, and self-help materials.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize resource databases
        self.crisis_resources = []
        self.professional_resources = []
        self.self_help_resources = []
        
        # Load resources
        self._load_default_resources()
        self._load_custom_resources()
        
        self.logger.info("Mental Health Resource Provider initialized")
    
    def _load_default_resources(self) -> None:
        """Load default mental health resources"""
        
        # Crisis Resources
        self.crisis_resources = [
            CrisisResource(
                name="National Suicide Prevention Lifeline",
                phone="988",
                text="Text 'HELLO' to 741741",
                website="https://suicidepreventionlifeline.org/",
                description="24/7 free and confidential support for people in distress",
                availability="24/7",
                languages=["English", "Spanish"],
                specialties=["suicide", "crisis", "emotional distress"]
            ),
            CrisisResource(
                name="Crisis Text Line",
                phone="",
                text="Text HOME to 741741",
                website="https://www.crisistextline.org/",
                description="Free, 24/7 support via text message",
                availability="24/7",
                languages=["English", "Spanish"],
                specialties=["crisis", "emotional support", "text-based"]
            ),
            CrisisResource(
                name="National Domestic Violence Hotline",
                phone="1-800-799-7233",
                text="Text START to 88788",
                website="https://www.thehotline.org/",
                description="24/7 support for domestic violence survivors",
                availability="24/7",
                languages=["English", "Spanish", "200+ languages via interpretation"],
                specialties=["domestic violence", "abuse", "safety planning"]
            ),
            CrisisResource(
                name="SAMHSA National Helpline",
                phone="1-800-662-4357",
                text="",
                website="https://www.samhsa.gov/find-help/national-helpline",
                description="Treatment referral and information service",
                availability="24/7",
                languages=["English", "Spanish"],
                specialties=["substance abuse", "mental health", "treatment referral"]
            ),
            CrisisResource(
                name="Trans Lifeline",
                phone="877-565-8860",
                text="",
                website="https://translifeline.org/",
                description="Crisis support for transgender community",
                availability="24/7",
                languages=["English"],
                specialties=["transgender", "LGBTQ+", "crisis support"]
            ),
            CrisisResource(
                name="Veterans Crisis Line",
                phone="1-800-273-8255",
                text="Text 838255",
                website="https://www.veteranscrisisline.net/",
                description="Crisis support for veterans and their families",
                availability="24/7",
                languages=["English"],
                specialties=["veterans", "military", "PTSD"]
            )
        ]
        
        # Professional Resources
        self.professional_resources = [
            ProfessionalResource(
                name="Psychology Today",
                type="therapist_directory",
                website="https://www.psychologytoday.com/us/therapists",
                description="Comprehensive directory of mental health professionals",
                search_features=["location", "specialty", "insurance", "therapy type"],
                cost_info="Varies by provider",
                insurance_accepted=True
            ),
            ProfessionalResource(
                name="SAMHSA Treatment Locator",
                type="treatment_facility_directory",
                website="https://findtreatment.samhsa.gov/",
                description="Find mental health and substance abuse treatment facilities",
                search_features=["location", "treatment type", "payment options"],
                cost_info="Varies, includes low-cost options",
                insurance_accepted=True
            ),
            ProfessionalResource(
                name="BetterHelp",
                type="online_therapy",
                website="https://www.betterhelp.com/",
                description="Online counseling and therapy platform",
                search_features=["specialty", "therapist matching", "scheduling"],
                cost_info="$60-90 per week",
                insurance_accepted=False
            ),
            ProfessionalResource(
                name="Talkspace",
                type="online_therapy",
                website="https://www.talkspace.com/",
                description="Online therapy with licensed therapists",
                search_features=["therapy type", "messaging", "video sessions"],
                cost_info="$69-109 per week",
                insurance_accepted=True
            ),
            ProfessionalResource(
                name="Open Path Collective",
                type="affordable_therapy",
                website="https://openpathcollective.org/",
                description="Affordable therapy network ($30-$60 per session)",
                search_features=["location", "sliding scale", "specialty"],
                cost_info="$30-60 per session",
                insurance_accepted=False
            )
        ]
        
        # Self-Help Resources
        self.self_help_resources = [
            SelfHelpResource(
                name="Headspace",
                type="app",
                url="https://www.headspace.com/",
                description="Meditation and mindfulness app",
                cost="freemium",
                rating=4.8,
                categories=["mindfulness", "meditation", "sleep", "anxiety"]
            ),
            SelfHelpResource(
                name="Calm",
                type="app",
                url="https://www.calm.com/",
                description="Sleep, meditation and relaxation app",
                cost="freemium",
                rating=4.7,
                categories=["sleep", "meditation", "relaxation", "stress"]
            ),
            SelfHelpResource(
                name="MindShift",
                type="app",
                url="https://www.anxietycanada.com/resources/mindshift-app/",
                description="Free app for anxiety management",
                cost="free",
                rating=4.5,
                categories=["anxiety", "CBT", "coping skills"]
            ),
            SelfHelpResource(
                name="Sanvello",
                type="app",
                url="https://www.sanvello.com/",
                description="Anxiety and depression support app",
                cost="freemium",
                rating=4.3,
                categories=["anxiety", "depression", "mood tracking", "CBT"]
            ),
            SelfHelpResource(
                name="NAMI (National Alliance on Mental Illness)",
                type="website",
                url="https://www.nami.org/",
                description="Mental health education and support resources",
                cost="free",
                rating=None,
                categories=["education", "support groups", "advocacy"]
            ),
            SelfHelpResource(
                name="Mental Health America",
                type="website",
                url="https://www.mhanational.org/",
                description="Mental health screening and resources",
                cost="free",
                rating=None,
                categories=["screening", "education", "advocacy"]
            )
        ]
    
    def _load_custom_resources(self) -> None:
        """Load custom resources from configuration"""
        try:
            custom_resources = self.config.get('custom_resources', {})
            
            # Load custom crisis resources
            for resource_data in custom_resources.get('crisis', []):
                resource = CrisisResource(**resource_data)
                self.crisis_resources.append(resource)
            
            # Load custom professional resources
            for resource_data in custom_resources.get('professional', []):
                resource = ProfessionalResource(**resource_data)
                self.professional_resources.append(resource)
            
            # Load custom self-help resources
            for resource_data in custom_resources.get('self_help', []):
                resource = SelfHelpResource(**resource_data)
                self.self_help_resources.append(resource)
            
            self.logger.info("Custom resources loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load custom resources: {e}")
    
    def get_crisis_resources(self, specialty: Optional[str] = None, 
                           language: Optional[str] = None) -> List[CrisisResource]:
        """
        Get crisis intervention resources
        
        Args:
            specialty: Filter by specialty (e.g., "suicide", "domestic violence")
            language: Filter by language support
            
        Returns:
            List of relevant crisis resources
        """
        resources = self.crisis_resources.copy()
        
        if specialty:
            resources = [r for r in resources if specialty.lower() in [s.lower() for s in r.specialties]]
        
        if language:
            resources = [r for r in resources if language.lower() in [l.lower() for l in r.languages]]
        
        return resources
    
    def get_professional_resources(self, resource_type: Optional[str] = None,
                                 insurance_required: Optional[bool] = None) -> List[ProfessionalResource]:
        """
        Get professional mental health resources
        
        Args:
            resource_type: Filter by type (e.g., "online_therapy", "therapist_directory")
            insurance_required: Filter by insurance acceptance
            
        Returns:
            List of relevant professional resources
        """
        resources = self.professional_resources.copy()
        
        if resource_type:
            resources = [r for r in resources if r.type == resource_type]
        
        if insurance_required is not None:
            resources = [r for r in resources if r.insurance_accepted == insurance_required]
        
        return resources
    
    def get_self_help_resources(self, category: Optional[str] = None,
                              cost_filter: Optional[str] = None) -> List[SelfHelpResource]:
        """
        Get self-help resources
        
        Args:
            category: Filter by category (e.g., "anxiety", "depression")
            cost_filter: Filter by cost ("free", "paid", "freemium")
            
        Returns:
            List of relevant self-help resources
        """
        resources = self.self_help_resources.copy()
        
        if category:
            resources = [r for r in resources if category.lower() in [c.lower() for c in r.categories]]
        
        if cost_filter:
            resources = [r for r in resources if r.cost == cost_filter]
        
        # Sort by rating (highest first)
        resources.sort(key=lambda x: x.rating or 0, reverse=True)
        
        return resources
    
    def get_personalized_recommendations(self, risk_assessment: Dict) -> Dict[str, List]:
        """
        Get personalized resource recommendations based on risk assessment
        
        Args:
            risk_assessment: Mental health risk assessment results
            
        Returns:
            Dictionary with categorized recommendations
        """
        recommendations = {
            'immediate': [],
            'professional': [],
            'self_help': [],
            'support': []
        }
        
        overall_risk = risk_assessment.get('overall_risk', 'low')
        anxiety_risk = risk_assessment.get('anxiety_risk', 'low')
        depression_risk = risk_assessment.get('depression_risk', 'low')
        crisis_indicators = risk_assessment.get('crisis_indicators', False)
        
        # Immediate/Crisis resources
        if crisis_indicators or overall_risk == 'crisis':
            recommendations['immediate'] = self.get_crisis_resources()
        elif overall_risk == 'high':
            recommendations['immediate'] = self.get_crisis_resources()[:3]  # Top 3 crisis resources
        
        # Professional resources
        if overall_risk in ['medium', 'high', 'crisis']:
            recommendations['professional'] = self.get_professional_resources()
        
        # Self-help resources based on specific risks
        if anxiety_risk in ['medium', 'high']:
            anxiety_resources = self.get_self_help_resources(category='anxiety')
            recommendations['self_help'].extend(anxiety_resources[:3])
        
        if depression_risk in ['medium', 'high']:
            depression_resources = self.get_self_help_resources(category='depression')
            recommendations['self_help'].extend(depression_resources[:3])
        
        # General mental health resources
        if overall_risk == 'low':
            general_resources = self.get_self_help_resources()
            recommendations['self_help'].extend(general_resources[:2])
        
        # Support resources (always include)
        recommendations['support'] = [
            resource for resource in self.self_help_resources 
            if 'support groups' in resource.categories or 'education' in resource.categories
        ][:2]
        
        # Remove duplicates
        for category in recommendations:
            seen = set()
            unique_resources = []
            for resource in recommendations[category]:
                if resource.name not in seen:
                    seen.add(resource.name)
                    unique_resources.append(resource)
            recommendations[category] = unique_resources
        
        return recommendations
    
    def format_resources_for_display(self, resources: List, resource_type: str) -> List[Dict]:
        """
        Format resources for display in UI
        
        Args:
            resources: List of resource objects
            resource_type: Type of resources ("crisis", "professional", "self_help")
            
        Returns:
            List of formatted resource dictionaries
        """
        formatted = []
        
        for resource in resources:
            if resource_type == "crisis":
                formatted.append({
                    'name': resource.name,
                    'contact': f"Phone: {resource.phone}" + (f", Text: {resource.text}" if resource.text else ""),
                    'website': resource.website,
                    'description': resource.description,
                    'availability': resource.availability,
                    'specialties': ', '.join(resource.specialties)
                })
            
            elif resource_type == "professional":
                formatted.append({
                    'name': resource.name,
                    'type': resource.type.replace('_', ' ').title(),
                    'website': resource.website,
                    'description': resource.description,
                    'cost': resource.cost_info,
                    'insurance': "Accepts insurance" if resource.insurance_accepted else "No insurance"
                })
            
            elif resource_type == "self_help":
                formatted.append({
                    'name': resource.name,
                    'type': resource.type.title(),
                    'url': resource.url,
                    'description': resource.description,
                    'cost': resource.cost.title(),
                    'rating': f"{resource.rating}/5.0" if resource.rating else "Not rated",
                    'categories': ', '.join(resource.categories)
                })
        
        return formatted
    
    def generate_resource_summary(self) -> Dict[str, int]:
        """Generate summary of available resources"""
        return {
            'crisis_resources': len(self.crisis_resources),
            'professional_resources': len(self.professional_resources),
            'self_help_resources': len(self.self_help_resources),
            'total_resources': len(self.crisis_resources) + len(self.professional_resources) + len(self.self_help_resources)
        }
    
    def search_resources(self, query: str) -> Dict[str, List]:
        """
        Search across all resources
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with matching resources by category
        """
        query_lower = query.lower()
        results = {
            'crisis': [],
            'professional': [],
            'self_help': []
        }
        
        # Search crisis resources
        for resource in self.crisis_resources:
            if (query_lower in resource.name.lower() or 
                query_lower in resource.description.lower() or
                any(query_lower in specialty.lower() for specialty in resource.specialties)):
                results['crisis'].append(resource)
        
        # Search professional resources
        for resource in self.professional_resources:
            if (query_lower in resource.name.lower() or 
                query_lower in resource.description.lower() or
                query_lower in resource.type.lower()):
                results['professional'].append(resource)
        
        # Search self-help resources
        for resource in self.self_help_resources:
            if (query_lower in resource.name.lower() or 
                query_lower in resource.description.lower() or
                any(query_lower in category.lower() for category in resource.categories)):
                results['self_help'].append(resource)
        
        return results
    
    def get_emergency_contacts(self) -> List[Dict[str, str]]:
        """Get emergency contact information"""
        return [
            {
                'name': 'Emergency Services',
                'number': '911',
                'description': 'For immediate life-threatening emergencies'
            },
            {
                'name': 'National Suicide Prevention Lifeline',
                'number': '988',
                'description': '24/7 crisis support and suicide prevention'
            },
            {
                'name': 'Crisis Text Line',
                'number': 'Text HOME to 741741',
                'description': '24/7 crisis support via text'
            }
        ]
