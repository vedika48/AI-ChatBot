"""
Services package initialization

This module provides service classes that implement the application's business logic.
Services are responsible for handling data processing, external API integrations,
and other operations that are not directly tied to HTTP request handling.
"""
from app.services.job_search import JobSearchService
from app.services.nlp import NLPService
from app.services.user_service import UserService

# Create service instances to be used throughout the application
job_search_service = JobSearchService()
nlp_service = NLPService()
user_service = UserService()

# Define what gets imported when using 'from app.services import *'
__all__ = [
    'job_search_service',
    'nlp_service',
    'user_service',
    'JobSearchService',
    'NLPService',
    'UserService'
]