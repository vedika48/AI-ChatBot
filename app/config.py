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
    
    # NLP settings
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_SEQUENCE_LENGTH = 128
    
    # Gender bias detection threshold
    BIAS_THRESHOLD = 0.6

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

# Dictionary of configuration environments
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}