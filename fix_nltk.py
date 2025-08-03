#!/usr/bin/env python3
"""
Fix NLTK data issues for Streamlit Cloud deployment
This script ensures all required NLTK data is downloaded
"""

import nltk
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_ssl_context():
    """Fix SSL context for NLTK downloads"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_packages():
    """Download all required NLTK packages with multiple attempts"""
    
    # Fix SSL context first
    fix_ssl_context()
    
    # List of required packages (newer versions first)
    packages = [
        'punkt_tab',
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger_eng',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker_tab',
        'maxent_ne_chunker',
        'words',
        'vader_lexicon',
        'brown'
    ]
    
    successful = []
    failed = []
    
    for package in packages:
        try:
            logger.info(f"Downloading {package}...")
            result = nltk.download(package, quiet=False)
            if result:
                successful.append(package)
                logger.info(f"✅ Successfully downloaded {package}")
            else:
                failed.append(package)
                logger.warning(f"⚠️ Download returned False for {package}")
        except Exception as e:
            failed.append(package)
            logger.error(f"❌ Failed to download {package}: {e}")
    
    logger.info(f"\nDownload Summary:")
    logger.info(f"✅ Successful: {len(successful)} packages")
    logger.info(f"❌ Failed: {len(failed)} packages")
    
    if successful:
        logger.info(f"Successful downloads: {successful}")
    if failed:
        logger.warning(f"Failed downloads: {failed}")
    
    return len(successful), len(failed)

def test_nltk_functionality():
    """Test if NLTK functions work correctly"""
    logger.info("\nTesting NLTK functionality...")
    
    test_text = "Hello world. This is a test sentence."
    
    # Test sentence tokenization
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(test_text)
        logger.info(f"✅ Sentence tokenization works: {len(sentences)} sentences")
    except Exception as e:
        logger.error(f"❌ Sentence tokenization failed: {e}")
    
    # Test word tokenization
    try:
        from nltk.tokenize import word_tokenize
        words = word_tokenize(test_text)
        logger.info(f"✅ Word tokenization works: {len(words)} words")
    except Exception as e:
        logger.error(f"❌ Word tokenization failed: {e}")
    
    # Test stopwords
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        logger.info(f"✅ Stopwords loaded: {len(stop_words)} words")
    except Exception as e:
        logger.error(f"❌ Stopwords failed: {e}")
    
    # Test POS tagging
    try:
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize
        words = word_tokenize(test_text)
        pos_tags = pos_tag(words)
        logger.info(f"✅ POS tagging works: {len(pos_tags)} tags")
    except Exception as e:
        logger.error(f"❌ POS tagging failed: {e}")

def main():
    """Main function to fix NLTK issues"""
    logger.info("🔧 NLTK Fix Script for Streamlit Cloud")
    logger.info("=" * 50)
    
    # Download packages
    successful, failed = download_nltk_packages()
    
    # Test functionality
    test_nltk_functionality()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 50)
    
    if failed == 0:
        logger.info("🎉 All NLTK packages downloaded successfully!")
        logger.info("✅ NLTK should work correctly now")
    else:
        logger.warning(f"⚠️ {failed} packages failed to download")
        logger.info("💡 The application will use fallback methods for failed packages")
    
    logger.info("\n🚀 You can now run the Streamlit app!")

if __name__ == "__main__":
    main()
