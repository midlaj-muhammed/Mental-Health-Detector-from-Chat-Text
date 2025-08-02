"""
Model Setup Script for Mental Health Detector
Downloads and initializes required models for the application.
"""

import os
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")

    # Handle both old and new NLTK data names
    nltk_downloads = [
        ('punkt', 'punkt_tab'),
        ('stopwords', 'stopwords'),
        ('wordnet', 'wordnet'),
        ('omw-1.4', 'omw-1.4'),
        ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'),
        ('maxent_ne_chunker', 'maxent_ne_chunker_tab'),
        ('words', 'words')
    ]

    for primary, fallback in tqdm(nltk_downloads, desc="NLTK Downloads"):
        try:
            nltk.download(primary, quiet=True)
            logger.info(f"Downloaded NLTK data: {primary}")
        except Exception as e:
            try:
                nltk.download(fallback, quiet=True)
                logger.info(f"Downloaded NLTK data: {fallback}")
            except Exception as e2:
                logger.warning(f"Failed to download NLTK data {primary}/{fallback}: {e2}")

def download_spacy_model():
    """Download spaCy English model"""
    logger.info("Downloading spaCy English model...")
    
    try:
        # Try to load the model first
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model already available")
    except OSError:
        logger.info("spaCy model not found, downloading...")
        os.system("python -m spacy download en_core_web_sm")
        logger.info("spaCy model downloaded successfully")

def download_transformer_models():
    """Download and cache transformer models"""
    logger.info("Downloading transformer models...")
    
    models = [
        {
            'name': 'j-hartmann/emotion-english-distilroberta-base',
            'description': 'Emotion detection model'
        },
        {
            'name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'description': 'Sentiment analysis model'
        }
    ]
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_info in models:
        model_name = model_info['name']
        description = model_info['description']
        
        logger.info(f"Downloading {description}: {model_name}")
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Downloaded tokenizer for {model_name}")
            
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Downloaded model for {model_name}")
            
            # Save locally (optional)
            local_path = models_dir / model_name.replace('/', '_')
            local_path.mkdir(exist_ok=True)
            
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            
            logger.info(f"Saved {description} to {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            continue

def check_system_requirements():
    """Check system requirements and dependencies"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        logger.error("PyTorch not found. Please install PyTorch.")
        return False
    
    # Check transformers
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not found. Please install transformers.")
        return False
    
    # Check other dependencies
    dependencies = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('nltk', 'nltk'),
        ('spacy', 'spacy'),
        ('textblob', 'textblob'),
        ('scikit-learn', 'sklearn'),
        ('cryptography', 'cryptography'),
        ('pyyaml', 'yaml')
    ]

    missing_deps = []
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            logger.info(f"✓ {dep_name} available")
        except ImportError:
            missing_deps.append(dep_name)
            logger.warning(f"✗ {dep_name} not found")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        "models",
        "logs",
        "data",
        "temp",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_models():
    """Test that models can be loaded and used"""
    logger.info("Testing model loading...")
    
    try:
        # Test emotion detection model
        from src.models.emotion_detector import EmotionDetector
        emotion_detector = EmotionDetector()
        emotion_detector.load_model()
        
        # Test with sample text
        test_result = emotion_detector.detect_emotion("I feel happy today!")
        logger.info(f"Emotion detection test: {test_result.emotion} (confidence: {test_result.confidence:.2f})")
        
        # Test sentiment analysis model
        from src.models.sentiment_analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_analyzer.load_model()
        
        # Test with sample text
        sentiment_result = sentiment_analyzer.analyze_sentiment("I feel happy today!")
        logger.info(f"Sentiment analysis test: {sentiment_result.sentiment} (confidence: {sentiment_result.confidence:.2f})")
        
        logger.info("✓ All models loaded and tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Mental Health Detector setup...")
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met. Please install missing dependencies.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Download spaCy model
    download_spacy_model()
    
    # Download transformer models
    download_transformer_models()
    
    # Test models
    if test_models():
        logger.info("✓ Setup completed successfully!")
        logger.info("You can now run the application with: streamlit run app.py")
    else:
        logger.error("Setup completed with errors. Some models may not work correctly.")
        sys.exit(1)

if __name__ == "__main__":
    main()
