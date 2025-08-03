"""
Text Processing Utilities for Mental Health Analysis
Advanced text preprocessing and feature extraction for mental health detection.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional, Set
import logging
import numpy as np
from dataclasses import dataclass
from collections import Counter

# Download required NLTK data with better error handling
def download_nltk_data():
    """Download NLTK data with fallback for different versions"""
    logger = logging.getLogger(__name__)

    # Prioritize newer NLTK data names
    downloads = [
        ('punkt_tab', 'punkt'),  # Newer version first
        ('stopwords', 'stopwords'),
        ('wordnet', 'wordnet'),
        ('omw-1.4', 'omw-1.4'),
        ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger'),
        ('maxent_ne_chunker_tab', 'maxent_ne_chunker'),
        ('words', 'words'),
        ('vader_lexicon', 'vader_lexicon')
    ]

    for primary, fallback in downloads:
        success = False
        try:
            result = nltk.download(primary, quiet=True)
            if result:
                success = True
        except Exception:
            try:
                result = nltk.download(fallback, quiet=True)
                if result:
                    success = True
            except Exception:
                pass

        if not success:
            logger.warning(f"Failed to download NLTK data: {primary}/{fallback}")

# Download data on import
try:
    download_nltk_data()
except Exception as e:
    logging.getLogger(__name__).warning(f"NLTK data download failed: {e}")

@dataclass
class TextFeatures:
    """Data class for extracted text features"""
    # Basic features
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    
    # Linguistic features
    pos_tags: Dict[str, int]
    named_entities: List[str]
    
    # Emotional features
    emotional_words: List[str]
    intensity_markers: List[str]
    
    # Mental health specific features
    first_person_pronouns: int
    negative_words: int
    positive_words: int
    
    # Readability and complexity
    readability_score: float
    lexical_diversity: float
    
    # Temporal and social features
    time_references: List[str]
    social_references: List[str]

class TextProcessor:
    """
    Advanced text processor for mental health analysis.
    Extracts linguistic, emotional, and psychological features from text.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model (if available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        # Mental health related word lists
        self.emotional_words = {
            'positive': [
                'happy', 'joy', 'excited', 'grateful', 'hopeful', 'confident',
                'peaceful', 'content', 'optimistic', 'proud', 'accomplished',
                'loved', 'blessed', 'amazing', 'wonderful', 'fantastic'
            ],
            'negative': [
                'sad', 'depressed', 'anxious', 'worried', 'scared', 'angry',
                'frustrated', 'hopeless', 'empty', 'numb', 'worthless',
                'terrible', 'awful', 'horrible', 'devastating', 'overwhelming'
            ],
            'anxiety': [
                'panic', 'nervous', 'tense', 'restless', 'uneasy', 'worried',
                'fearful', 'apprehensive', 'jittery', 'on edge'
            ],
            'depression': [
                'sad', 'empty', 'hopeless', 'worthless', 'guilty', 'tired',
                'exhausted', 'numb', 'disconnected', 'isolated'
            ]
        }
        
        # Intensity markers
        self.intensity_markers = [
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'really', 'so', 'quite', 'rather', 'pretty',
            'always', 'never', 'everything', 'nothing', 'all', 'none'
        ]
        
        # First person pronouns
        self.first_person_pronouns = [
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        ]
        
        # Time references
        self.time_references = [
            'today', 'yesterday', 'tomorrow', 'now', 'then', 'always', 'never',
            'recently', 'lately', 'soon', 'future', 'past', 'present'
        ]
        
        # Social references
        self.social_references = [
            'family', 'friends', 'people', 'everyone', 'someone', 'nobody',
            'alone', 'together', 'social', 'relationship', 'partner'
        ]
    
    def preprocess_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Preprocess text while preserving important features for mental health analysis
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Preserve important punctuation patterns
        # Keep ellipsis, multiple exclamation marks, etc. as they indicate emotional state
        text = re.sub(r'\.{3,}', ' ELLIPSIS ', text)
        text = re.sub(r'!{2,}', ' MULTIPLE_EXCLAMATION ', text)
        text = re.sub(r'\?{2,}', ' MULTIPLE_QUESTION ', text)
        
        # Handle contractions (preserve emotional context)
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        # Remove URLs but keep emotional context
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Handle case conversion
        if not preserve_case:
            text = text.lower()
        
        return text
    
    def extract_features(self, text: str) -> TextFeatures:
        """
        Extract comprehensive features from text for mental health analysis
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextFeatures object with extracted features
        """
        if not text:
            return self._create_empty_features()
        
        # Preprocess text
        processed_text = self.preprocess_text(text, preserve_case=True)
        clean_text = self.preprocess_text(text, preserve_case=False)
        
        # Basic features with error handling
        try:
            words = word_tokenize(clean_text)
        except Exception as e:
            # Fallback to simple split if NLTK tokenizer fails
            words = clean_text.split()
            logger.warning(f"NLTK word tokenizer failed, using simple split: {e}")

        try:
            sentences = sent_tokenize(processed_text)
        except Exception as e:
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in processed_text.split('.') if s.strip()]
            logger.warning(f"NLTK sentence tokenizer failed, using simple split: {e}")
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # POS tagging
        pos_tags = self._extract_pos_features(words)
        
        # Named entities
        named_entities = self._extract_named_entities(processed_text)
        
        # Emotional features
        emotional_words = self._extract_emotional_words(clean_text)
        intensity_markers = self._extract_intensity_markers(words)
        
        # Mental health specific features
        first_person_count = self._count_first_person_pronouns(words)
        negative_count = self._count_negative_words(words)
        positive_count = self._count_positive_words(words)
        
        # Readability and complexity
        readability_score = self._calculate_readability(processed_text)
        lexical_diversity = self._calculate_lexical_diversity(words)
        
        # Temporal and social features
        time_refs = self._extract_time_references(words)
        social_refs = self._extract_social_references(words)
        
        return TextFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            pos_tags=pos_tags,
            named_entities=named_entities,
            emotional_words=emotional_words,
            intensity_markers=intensity_markers,
            first_person_pronouns=first_person_count,
            negative_words=negative_count,
            positive_words=positive_count,
            readability_score=readability_score,
            lexical_diversity=lexical_diversity,
            time_references=time_refs,
            social_references=social_refs
        )
    
    def _extract_pos_features(self, words: List[str]) -> Dict[str, int]:
        """Extract part-of-speech features"""
        pos_tags = pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        # Group into meaningful categories
        pos_features = {
            'nouns': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                    pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0),
            'verbs': pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                    pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                    pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),
            'adjectives': pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                         pos_counts.get('JJS', 0),
            'adverbs': pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                      pos_counts.get('RBS', 0),
            'pronouns': pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)
        }
        
        return pos_features
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities using spaCy if available"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents]
        else:
            # Fallback to NLTK with error handling
            try:
                tokens = word_tokenize(text)
                pos_tags_list = pos_tag(tokens)
                chunks = ne_chunk(pos_tags_list)
            except Exception as e:
                logger.warning(f"NLTK NER processing failed: {e}")
                # Return empty list if NLTK fails
                return []
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entities.append(' '.join([token for token, pos in chunk.leaves()]))
        
        return entities
    
    def _extract_emotional_words(self, text: str) -> List[str]:
        """Extract emotional words from text"""
        words = text.split()
        emotional_words = []
        
        for category, word_list in self.emotional_words.items():
            for word in word_list:
                if word in words:
                    emotional_words.append(f"{category}:{word}")
        
        return emotional_words
    
    def _extract_intensity_markers(self, words: List[str]) -> List[str]:
        """Extract intensity markers"""
        return [word for word in words if word.lower() in self.intensity_markers]
    
    def _count_first_person_pronouns(self, words: List[str]) -> int:
        """Count first person pronouns"""
        return sum(1 for word in words if word.lower() in self.first_person_pronouns)
    
    def _count_negative_words(self, words: List[str]) -> int:
        """Count negative emotional words"""
        negative_words = self.emotional_words['negative'] + self.emotional_words['anxiety'] + \
                        self.emotional_words['depression']
        return sum(1 for word in words if word.lower() in negative_words)
    
    def _count_positive_words(self, words: List[str]) -> int:
        """Count positive emotional words"""
        return sum(1 for word in words if word.lower() in self.emotional_words['positive'])
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using TextBlob"""
        try:
            blob = TextBlob(text)
            sentences = len(blob.sentences)
            words = len(blob.words)
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Simple readability approximation
            avg_sentence_length = words / sentences
            readability = max(0, min(100, 100 - (avg_sentence_length - 15) * 2))
            
            return readability
        except:
            return 0.0
    
    def _calculate_lexical_diversity(self, words: List[str]) -> float:
        """Calculate lexical diversity (type-token ratio)"""
        if not words:
            return 0.0
        
        unique_words = set(word.lower() for word in words if word.isalpha())
        total_words = len([word for word in words if word.isalpha()])
        
        if total_words == 0:
            return 0.0
        
        return len(unique_words) / total_words
    
    def _extract_time_references(self, words: List[str]) -> List[str]:
        """Extract time-related references"""
        return [word for word in words if word.lower() in self.time_references]
    
    def _extract_social_references(self, words: List[str]) -> List[str]:
        """Extract social-related references"""
        return [word for word in words if word.lower() in self.social_references]
    
    def _create_empty_features(self) -> TextFeatures:
        """Create empty features object"""
        return TextFeatures(
            word_count=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            pos_tags={},
            named_entities=[],
            emotional_words=[],
            intensity_markers=[],
            first_person_pronouns=0,
            negative_words=0,
            positive_words=0,
            readability_score=0.0,
            lexical_diversity=0.0,
            time_references=[],
            social_references=[]
        )
    
    def analyze_text_patterns(self, text: str) -> Dict[str, any]:
        """
        Analyze text patterns specific to mental health indicators
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with pattern analysis results
        """
        features = self.extract_features(text)
        
        # Calculate mental health indicators
        indicators = {
            'emotional_intensity': len(features.intensity_markers) / max(features.word_count, 1),
            'self_focus': features.first_person_pronouns / max(features.word_count, 1),
            'negative_sentiment_ratio': features.negative_words / max(features.positive_words + features.negative_words, 1),
            'social_isolation_indicators': len([ref for ref in features.social_references if ref in ['alone', 'nobody', 'isolated']]),
            'temporal_focus': {
                'past_focus': len([ref for ref in features.time_references if ref in ['yesterday', 'past', 'then']]),
                'present_focus': len([ref for ref in features.time_references if ref in ['now', 'today', 'present']]),
                'future_focus': len([ref for ref in features.time_references if ref in ['tomorrow', 'future', 'soon']])
            },
            'linguistic_complexity': features.lexical_diversity,
            'readability': features.readability_score
        }
        
        return {
            'features': features,
            'indicators': indicators,
            'summary': self._generate_pattern_summary(indicators)
        }
    
    def _generate_pattern_summary(self, indicators: Dict) -> Dict[str, str]:
        """Generate summary of text patterns"""
        summary = {}
        
        # Emotional intensity
        if indicators['emotional_intensity'] > 0.1:
            summary['emotional_state'] = 'high_intensity'
        elif indicators['emotional_intensity'] > 0.05:
            summary['emotional_state'] = 'moderate_intensity'
        else:
            summary['emotional_state'] = 'low_intensity'
        
        # Self-focus
        if indicators['self_focus'] > 0.15:
            summary['self_focus'] = 'high'
        elif indicators['self_focus'] > 0.08:
            summary['self_focus'] = 'moderate'
        else:
            summary['self_focus'] = 'low'
        
        # Sentiment balance
        if indicators['negative_sentiment_ratio'] > 0.7:
            summary['sentiment_balance'] = 'predominantly_negative'
        elif indicators['negative_sentiment_ratio'] > 0.3:
            summary['sentiment_balance'] = 'mixed'
        else:
            summary['sentiment_balance'] = 'predominantly_positive'
        
        return summary
