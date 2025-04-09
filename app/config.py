import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-development')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API keys for job search services
    INDEED_API_KEY = os.environ.get('INDEED_API_KEY')
    LINKEDIN_API_KEY = os.environ.get('LINKEDIN_API_KEY')
    GLASSDOOR_API_KEY = os.environ.get('GLASSDOOR_API_KEY')
    MONSTER_API_KEY = os.environ.get('MONSTER_API_KEY')
    ZIPRECRUITER_API_KEY = os.environ.get('ZIPRECRUITER_API_KEY')
    DICE_API_KEY = os.environ.get('DICE_API_KEY')
    CAREERBUILDER_API_KEY = os.environ.get('CAREERBUILDER_API_KEY')
    SIMPLYHIRED_API_KEY = os.environ.get('SIMPLYHIRED_API_KEY')
    
    # Enhanced NLP settings
    MODEL_NAME = os.environ.get('NLP_MODEL_NAME', 'bert-base-uncased')  # Upgraded from distilbert
    MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH', 256))  # Increased from 128
    INTENT_CONFIDENCE_THRESHOLD = float(os.environ.get('INTENT_CONFIDENCE_THRESHOLD', 0.7))
    FALLBACK_INTENT = 'help'
    
    # Fine-tuning configuration
    FINE_TUNING_EPOCHS = int(os.environ.get('FINE_TUNING_EPOCHS', 3))
    FINE_TUNING_BATCH_SIZE = int(os.environ.get('FINE_TUNING_BATCH_SIZE', 16))
    FINE_TUNING_LEARNING_RATE = float(os.environ.get('FINE_TUNING_LEARNING_RATE', 5e-5))
    TRAINING_DATA_PATH = os.environ.get('TRAINING_DATA_PATH', 'data/training.csv')
    
    # Model caching settings
    ENABLE_MODEL_CACHING = os.environ.get('ENABLE_MODEL_CACHING', 'True').lower() == 'true'
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', 100))
    
    # Language settings
    DEFAULT_LANGUAGE = os.environ.get('DEFAULT_LANGUAGE', 'en')
    SUPPORTED_LANGUAGES = os.environ.get('SUPPORTED_LANGUAGES', 'en,es,fr').split(',')
    
    # Gender bias detection settings
    BIAS_THRESHOLD = float(os.environ.get('BIAS_THRESHOLD', 0.6))
    ENABLE_BIAS_DETECTION = os.environ.get('ENABLE_BIAS_DETECTION', 'True').lower() == 'true'
    
    # Response enhancement settings
    USE_TEMPLATE_AUGMENTATION = os.environ.get('USE_TEMPLATE_AUGMENTATION', 'True').lower() == 'true'
    RESPONSE_PERSONALIZATION_LEVEL = int(os.environ.get('RESPONSE_PERSONALIZATION_LEVEL', 2))  # 0-3 scale
    
    # Entity extraction settings
    ENTITY_CONFIDENCE_THRESHOLD = float(os.environ.get('ENTITY_CONFIDENCE_THRESHOLD', 0.65))
    EXTRACT_CUSTOM_ENTITIES = os.environ.get('EXTRACT_CUSTOM_ENTITIES', 'True').lower() == 'true'
    CUSTOM_ENTITIES_FILE = os.environ.get('CUSTOM_ENTITIES_FILE', 'data/custom_entities.json')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')
    ENABLE_PERFORMANCE_TRACKING = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    ENABLE_PERFORMANCE_TRACKING = os.environ.get('ENABLE_PERFORMANCE_TRACKING', 'False').lower() == 'true'
    
    # More secure settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PREFERRED_URL_SCHEME = 'https'
    
    # Performance optimization
    MODEL_BATCH_PROCESSING = True
    
class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    
# Dictionary of configuration environments
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}