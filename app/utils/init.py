"""
Utils package initialization

This module provides utility functions and helpers used throughout the application.
These are general-purpose tools that don't fit into specific domains or services.
"""
from app.utils.helpers import (
    format_date,
    parse_salary,
    standardize_job_title,
    extract_gender_inclusive_terms,
    calculate_gender_pay_gap,
    find_inclusive_companies
)

# Define what gets imported when using 'from app.utils import *'
__all__ = [
    'format_date',
    'parse_salary',
    'standardize_job_title',
    'extract_gender_inclusive_terms',
    'calculate_gender_pay_gap',
    'find_inclusive_companies'
]