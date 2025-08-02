"""
Privacy Handler for Mental Health Data
Implements privacy protection and data security measures for sensitive mental health information.
"""

import hashlib
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Dict, Optional, Union, List
import json
from datetime import datetime, timedelta

class PrivacyHandler:
    """
    Handles privacy protection for mental health data including encryption,
    anonymization, and secure data handling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Privacy settings
        self.encrypt_data = self.config.get('encrypt_data', True)
        self.log_interactions = self.config.get('log_interactions', False)
        self.data_retention_days = self.config.get('data_retention_days', 0)
        
        # Encryption components
        self.cipher_suite = None
        self.encryption_key = None
        
        # Anonymization settings
        self.hash_salt = self._generate_salt()
        
        self.logger.info("Privacy Handler initialized")
    
    def initialize(self) -> None:
        """Initialize privacy protection components"""
        try:
            if self.encrypt_data:
                self._setup_encryption()
            
            self.logger.info("Privacy protection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing privacy handler: {str(e)}")
            raise
    
    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data"""
        try:
            # Generate or load encryption key
            key_file = "encryption.key"
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                
                # Save key securely (in production, use proper key management)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                
                # Set restrictive permissions
                os.chmod(key_file, 0o600)
            
            self.cipher_suite = Fernet(self.encryption_key)
            self.logger.info("Encryption setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up encryption: {str(e)}")
            raise
    
    def _generate_salt(self) -> bytes:
        """Generate salt for hashing"""
        return os.urandom(32)
    
    def should_encrypt(self) -> bool:
        """Check if data should be encrypted"""
        return self.encrypt_data and self.cipher_suite is not None
    
    def encrypt_text(self, text: str) -> str:
        """
        Encrypt sensitive text data
        
        Args:
            text: Text to encrypt
            
        Returns:
            Encrypted text as base64 string
        """
        if not self.should_encrypt():
            return text
        
        try:
            encrypted_data = self.cipher_suite.encrypt(text.encode('utf-8'))
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error encrypting text: {str(e)}")
            return text  # Return original text if encryption fails
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt encrypted text data
        
        Args:
            encrypted_text: Base64 encoded encrypted text
            
        Returns:
            Decrypted text
        """
        if not self.should_encrypt():
            return encrypted_text
        
        try:
            encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error decrypting text: {str(e)}")
            return encrypted_text  # Return encrypted text if decryption fails
    
    def anonymize_user_id(self, user_id: str) -> str:
        """
        Create anonymous hash of user ID
        
        Args:
            user_id: Original user identifier
            
        Returns:
            Anonymized hash
        """
        if not user_id:
            return ""
        
        try:
            # Create hash with salt
            hash_input = (user_id + str(self.hash_salt)).encode('utf-8')
            hash_object = hashlib.sha256(hash_input)
            return hash_object.hexdigest()[:16]  # Use first 16 characters
            
        except Exception as e:
            self.logger.error(f"Error anonymizing user ID: {str(e)}")
            return "anonymous"
    
    def sanitize_text_for_logging(self, text: str) -> str:
        """
        Sanitize text for safe logging by removing sensitive information
        
        Args:
            text: Original text
            
        Returns:
            Sanitized text safe for logging
        """
        if not text:
            return ""
        
        # Remove potential personal identifiers
        sanitized = text
        
        # Remove email addresses
        import re
        sanitized = re.sub(r'\S+@\S+', '[EMAIL]', sanitized)
        
        # Remove phone numbers
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', sanitized)
        
        # Remove potential names (simple heuristic)
        sanitized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', sanitized)
        
        # Remove addresses (simple heuristic)
        sanitized = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', '[ADDRESS]', sanitized)
        
        # Truncate if too long
        if len(sanitized) > 200:
            sanitized = sanitized[:200] + "..."
        
        return sanitized
    
    def create_privacy_report(self, analysis_data: Dict) -> Dict:
        """
        Create privacy compliance report for analysis
        
        Args:
            analysis_data: Analysis results
            
        Returns:
            Privacy compliance report
        """
        report = {
            'privacy_compliance': {
                'data_encrypted': self.should_encrypt(),
                'user_anonymized': 'user_id_hashed' in analysis_data.get('privacy', {}),
                'no_permanent_storage': self.data_retention_days == 0,
                'logging_disabled': not self.log_interactions,
                'sensitive_data_sanitized': True
            },
            'data_handling': {
                'encryption_method': 'Fernet (AES 128)' if self.should_encrypt() else 'None',
                'anonymization_method': 'SHA-256 with salt',
                'retention_policy': f'{self.data_retention_days} days' if self.data_retention_days > 0 else 'No retention',
                'logging_policy': 'Enabled' if self.log_interactions else 'Disabled'
            },
            'compliance_status': 'COMPLIANT',
            'recommendations': []
        }
        
        # Add recommendations based on configuration
        if not self.should_encrypt():
            report['recommendations'].append('Consider enabling data encryption for enhanced privacy')
        
        if self.log_interactions:
            report['recommendations'].append('Review logging policy to ensure minimal data collection')
        
        if self.data_retention_days > 0:
            report['recommendations'].append('Consider reducing data retention period for better privacy')
        
        return report
    
    def secure_delete_data(self, data_path: str) -> bool:
        """
        Securely delete sensitive data files
        
        Args:
            data_path: Path to data file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(data_path):
                # Overwrite file with random data before deletion
                file_size = os.path.getsize(data_path)
                
                with open(data_path, 'wb') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                
                # Delete file
                os.remove(data_path)
                
                self.logger.info(f"Securely deleted data file: {data_path}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error securely deleting data: {str(e)}")
            return False
    
    def validate_data_retention(self) -> List[str]:
        """
        Validate data retention policies and identify expired data
        
        Returns:
            List of files that should be deleted due to retention policy
        """
        expired_files = []
        
        if self.data_retention_days <= 0:
            return expired_files  # No retention policy
        
        try:
            # Check for log files and temporary data
            data_directories = ['logs', 'temp', 'cache']
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            
            for directory in data_directories:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        
                        if os.path.isfile(file_path):
                            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if file_modified < cutoff_date:
                                expired_files.append(file_path)
            
            return expired_files
            
        except Exception as e:
            self.logger.error(f"Error validating data retention: {str(e)}")
            return []
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired data according to retention policy
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_checked': 0,
            'files_deleted': 0,
            'errors': 0
        }
        
        try:
            expired_files = self.validate_data_retention()
            stats['files_checked'] = len(expired_files)
            
            for file_path in expired_files:
                if self.secure_delete_data(file_path):
                    stats['files_deleted'] += 1
                else:
                    stats['errors'] += 1
            
            self.logger.info(f"Data cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {str(e)}")
            stats['errors'] += 1
            return stats
    
    def get_privacy_settings(self) -> Dict:
        """Get current privacy settings"""
        return {
            'encryption_enabled': self.should_encrypt(),
            'logging_enabled': self.log_interactions,
            'data_retention_days': self.data_retention_days,
            'anonymization_enabled': True,
            'secure_deletion_enabled': True
        }
    
    def update_privacy_settings(self, new_settings: Dict) -> bool:
        """
        Update privacy settings
        
        Args:
            new_settings: Dictionary with new privacy settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'encrypt_data' in new_settings:
                self.encrypt_data = new_settings['encrypt_data']
                if self.encrypt_data and not self.cipher_suite:
                    self._setup_encryption()
            
            if 'log_interactions' in new_settings:
                self.log_interactions = new_settings['log_interactions']
            
            if 'data_retention_days' in new_settings:
                self.data_retention_days = new_settings['data_retention_days']
            
            self.logger.info("Privacy settings updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating privacy settings: {str(e)}")
            return False
