"""
Tests package initialization

This module defines the test package and makes test classes available for discovery
by test runners. It also provides common fixtures and utilities for testing.
"""
import os
import sys

# Add the project root directory to the Python path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test classes to make them available for test discovery
from tests.test_services import TestJobSearchService, TestNLPService, TestUserService

# Import any test fixtures or helpers that should be available across test modules
from tests.fixtures import (
    app_fixture,
    db_fixture,
    test_user_fixture,
    test_job_fixture
)

# Define what gets imported when using 'from tests import *'
__all__ = [
    'TestJobSearchService',
    'TestNLPService',
    'TestUserService',
    'app_fixture',
    'db_fixture',
    'test_user_fixture',
    'test_job_fixture'
]