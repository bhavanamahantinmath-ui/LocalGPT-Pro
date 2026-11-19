"""
Firebase Client Configuration loader for LocalGPT Pro.
Loads web credentials from .env using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

def get_firebase_web_config() -> dict:
    """
    Returns the Firebase Web SDK configuration dictionary.
    """
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyDemoDummyKeyForLocalTesting123456"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "localgpt-pro.firebaseapp.com"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "localgpt-pro"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "localgpt-pro.appspot.com"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "1234567890"),
        "appId": os.getenv("FIREBASE_APP_ID", "1:1234567890:web:abcdef1234567890")
    }

def is_firebase_configured() -> bool:
    """Checks if real Firebase web credentials have been supplied."""
    api_key = os.getenv("FIREBASE_API_KEY", "")
    return bool(api_key and not api_key.startswith("AIzaSyDemo"))

