"""
Models package initialization

This module imports all models to make them available when importing from the models package.
It also helps with the SQLAlchemy model registry.
"""
# Import models so they are registered with SQLAlchemy
from app.models.user import User
from app.models.job import Job

# Define what gets imported when using 'from app.models import *'
__all__ = ['User', 'Job']