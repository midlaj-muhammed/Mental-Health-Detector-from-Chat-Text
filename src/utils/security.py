"""
Security Module for Mental Health Detector
Implements comprehensive security measures for protecting sensitive mental health data.
"""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityManager:
    """
    Comprehensive security manager for mental health data protection.
    Implements encryption, access control, audit logging, and data sanitization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Security settings
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        self.audit_logging = self.config.get('audit_logging', True)
        self.session_timeout = self.config.get('session_timeout_minutes', 30)
        
        # Encryption components
        self.symmetric_key = None
        self.cipher_suite = None
        
        # Session management
        self.active_sessions = {}
        self.session_keys = {}
        
        # Audit trail
        self.audit_log = []
        
        # Security policies
        self.security_policies = {
            'min_password_length': 12,
            'require_special_chars': True,
            'max_failed_attempts': 3,
            'lockout_duration_minutes': 15,
            'data_retention_days': 0,  # No retention by default
            'require_2fa': False
        }
        
        self.logger.info("Security Manager initialized")
    
    def initialize_security(self) -> bool:
        """Initialize security components"""
        try:
            if self.encryption_enabled:
                self._setup_encryption()
            
            self._setup_audit_logging()
            self._load_security_policies()
            
            self.logger.info("Security components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security: {e}")
            return False
    
    def _setup_encryption(self) -> None:
        """Setup encryption keys and cipher"""
        try:
            key_file = Path("security") / "encryption.key"
            key_file.parent.mkdir(exist_ok=True)
            
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.symmetric_key = f.read()
            else:
                # Generate new key
                self.symmetric_key = Fernet.generate_key()
                
                # Save key securely
                with open(key_file, 'wb') as f:
                    f.write(self.symmetric_key)
                
                # Set restrictive permissions
                os.chmod(key_file, 0o600)
            
            self.cipher_suite = Fernet(self.symmetric_key)
            self.logger.info("Encryption setup completed")
            
        except Exception as e:
            self.logger.error(f"Encryption setup failed: {e}")
            raise
    
    def _setup_audit_logging(self) -> None:
        """Setup audit logging"""
        if self.audit_logging:
            audit_dir = Path("logs") / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up audit log file
            self.audit_log_file = audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
            
            self.logger.info("Audit logging setup completed")
    
    def _load_security_policies(self) -> None:
        """Load security policies from configuration"""
        config_policies = self.config.get('security_policies', {})
        self.security_policies.update(config_policies)
        
        self.logger.info("Security policies loaded")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data: Sensitive data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        if not self.encryption_enabled or not self.cipher_suite:
            return data
        
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            self._log_security_event('encryption_failure', {'error': str(e)})
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.encryption_enabled or not self.cipher_suite:
            return encrypted_data
        
        try:
            decoded_data = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            self._log_security_event('decryption_failure', {'error': str(e)})
            return encrypted_data
    
    def sanitize_text_data(self, text: str) -> str:
        """
        Sanitize text data by removing or masking sensitive information
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        sanitized = text
        
        # Remove/mask personal identifiers
        import re
        
        # Email addresses
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                          '[EMAIL_REDACTED]', sanitized)
        
        # Phone numbers
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
                          '[PHONE_REDACTED]', sanitized)
        
        # Social Security Numbers
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 
                          '[SSN_REDACTED]', sanitized)
        
        # Credit card numbers (basic pattern)
        sanitized = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                          '[CARD_REDACTED]', sanitized)
        
        # Addresses (basic pattern)
        sanitized = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', 
                          '[ADDRESS_REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Names (simple heuristic - capitalized words)
        sanitized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 
                          '[NAME_REDACTED]', sanitized)
        
        return sanitized
    
    def create_secure_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a secure session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session token
        """
        try:
            # Generate secure session token
            session_token = secrets.token_urlsafe(32)
            
            # Create session data
            session_data = {
                'token': session_token,
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=self.session_timeout)
            }
            
            # Store session
            self.active_sessions[session_token] = session_data
            
            # Generate session key for additional security
            session_key = Fernet.generate_key()
            self.session_keys[session_token] = session_key
            
            self._log_security_event('session_created', {
                'session_token': session_token[:8] + '...',  # Log only partial token
                'user_id': user_id
            })
            
            return session_token
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            self._log_security_event('session_creation_failure', {'error': str(e)})
            raise
    
    def validate_session(self, session_token: str) -> bool:
        """
        Validate a session token
        
        Args:
            session_token: Session token to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            if session_token not in self.active_sessions:
                self._log_security_event('invalid_session_access', {
                    'session_token': session_token[:8] + '...'
                })
                return False
            
            session_data = self.active_sessions[session_token]
            
            # Check expiration
            if datetime.now() > session_data['expires_at']:
                self._cleanup_expired_session(session_token)
                self._log_security_event('session_expired', {
                    'session_token': session_token[:8] + '...'
                })
                return False
            
            # Update last activity
            session_data['last_activity'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return False
    
    def _cleanup_expired_session(self, session_token: str) -> None:
        """Clean up expired session"""
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
        
        if session_token in self.session_keys:
            del self.session_keys[session_token]
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        expired_sessions = []
        
        for token, session_data in self.active_sessions.items():
            if current_time > session_data['expires_at']:
                expired_sessions.append(token)
        
        for token in expired_sessions:
            self._cleanup_expired_session(token)
        
        if expired_sessions:
            self._log_security_event('expired_sessions_cleanup', {
                'count': len(expired_sessions)
            })
        
        return len(expired_sessions)
    
    def hash_sensitive_identifier(self, identifier: str, salt: Optional[bytes] = None) -> str:
        """
        Create a secure hash of sensitive identifier
        
        Args:
            identifier: Identifier to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Hashed identifier
        """
        if not salt:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(identifier.encode('utf-8'))
        return base64.b64encode(salt + key).decode('utf-8')
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event for audit trail"""
        if not self.audit_logging:
            return
        
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': details,
                'source': 'SecurityManager'
            }
            
            # Add to in-memory audit log
            self.audit_log.append(event)
            
            # Write to audit log file
            if hasattr(self, 'audit_log_file'):
                with open(self.audit_log_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
            
            # Keep only recent events in memory
            if len(self.audit_log) > 1000:
                self.audit_log = self.audit_log[-500:]
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            'encryption_enabled': self.encryption_enabled,
            'audit_logging_enabled': self.audit_logging,
            'active_sessions': len(self.active_sessions),
            'security_policies': self.security_policies,
            'recent_events': len(self.audit_log),
            'last_cleanup': datetime.now().isoformat()
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        # Clean up expired sessions first
        expired_count = self.cleanup_expired_sessions()
        
        # Analyze recent security events
        recent_events = self.audit_log[-100:] if self.audit_log else []
        event_types = {}
        
        for event in recent_events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'security_status': self.get_security_status(),
            'session_management': {
                'active_sessions': len(self.active_sessions),
                'expired_sessions_cleaned': expired_count,
                'session_timeout_minutes': self.session_timeout
            },
            'audit_summary': {
                'total_events': len(self.audit_log),
                'recent_events': len(recent_events),
                'event_types': event_types
            },
            'recommendations': self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if not self.encryption_enabled:
            recommendations.append("Enable encryption for sensitive data protection")
        
        if not self.audit_logging:
            recommendations.append("Enable audit logging for security monitoring")
        
        if len(self.active_sessions) > 100:
            recommendations.append("Monitor high number of active sessions")
        
        if self.session_timeout > 60:
            recommendations.append("Consider reducing session timeout for better security")
        
        recommendations.extend([
            "Regularly review audit logs for suspicious activity",
            "Implement regular security assessments",
            "Keep security policies up to date",
            "Monitor for data breaches and unauthorized access"
        ])
        
        return recommendations
    
    def secure_delete_data(self, file_path: str) -> bool:
        """
        Securely delete sensitive data files
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return True
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite with random data multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # Multiple passes
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            os.remove(file_path)
            
            self._log_security_event('secure_file_deletion', {
                'file_path': file_path,
                'file_size': file_size
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Secure deletion failed: {e}")
            self._log_security_event('secure_deletion_failure', {
                'file_path': file_path,
                'error': str(e)
            })
            return False
