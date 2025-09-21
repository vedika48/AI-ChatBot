import pandas as pd
import numpy as np
from datetime import datetime

class SalaryDataService:
    def __init__(self):
        # Sample salary data - in production, this would come from a database
        self.salary_data = self._load_sample_salary_data()

    def _load_sample_salary_data(self):
        """Load sample salary data for Indian market"""
        return {
            'software_engineer': {
                'bangalore': {
                    '0-2': {'min': 400000, 'median': 800000, 'max': 1500000},
                    '2-5': {'min': 800000, 'median': 1500000, 'max': 2500000},
                    '5-8': {'min': 1500000, 'median': 2500000, 'max': 4000000},
                    '8+': {'min': 2500000, 'median': 4000000, 'max': 8000000}
                },
                'mumbai': {
                    '0-2': {'min': 450000, 'median': 850000, 'max': 1600000},
                    '2-5': {'min': 850000, 'median': 1600000, 'max': 2700000},
                    '5-8': {'min': 1600000, 'median': 2700000, 'max': 4200000},
                    '8+': {'min': 2700000, 'median': 4200000, 'max': 8500000}
                }
            },
            'data_scientist': {
                'bangalore': {
                    '0-2': {'min': 600000, 'median': 1000000, 'max': 1800000},
                    '2-5': {'min': 1000000, 'median': 1800000, 'max': 3000000},
                    '5-8': {'min': 1800000, 'median': 3000000, 'max': 5000000},
                    '8+': {'min': 3000000, 'median': 5000000, 'max': 10000000}
                }
            }
        }

    def get_market_data(self, role, location, experience):
        """Get market salary data for role, location, and experience"""
        try:
            exp_range = self._get_experience_range(experience)
            role_data = self.salary_data.get(role.lower().replace(' ', '_'), {})
            location_data = role_data.get(location.lower(), {})
            salary_data = location_data.get(exp_range, {})
            
            if not salary_data:
                # Fallback to default data
                salary_data = {'min': 500000, 'median': 1200000, 'max': 2500000}
            
            return {
                'role': role,
                'location': location,
                'experience_range': exp_range,
                'salary_range': salary_data,
                'percentile_25': int(salary_data['min'] * 1.2),
                'percentile_50': salary_data['median'],
                'percentile_75': int(salary_data['max'] * 0.8),
                'percentile_90': salary_data['max'],
                'currency': 'INR',
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            return self._get_default_market_data()

    def get_negotiation_tips(self, current_salary, target_salary, user_profile):
        """Generate personalized negotiation tips"""
        tips = []
        
        if target_salary > current_salary * 1.5:
            tips.append("Your target salary is significantly higher than current. Focus on demonstrating increased value and skills gained.")
        
        if user_profile.get('experience_years', 0) < 3:
            tips.append("For early career, emphasize learning potential and fresh perspectives rather than just experience.")
        
        location = user_profile.get('location', '').lower()
        if location in ['bangalore', 'mumbai']:
            tips.append(f"{location.title()} has higher living costs. Use cost of living data to justify higher salary expectations.")
        
        tips.extend([
            "Research the company's recent funding, growth, and compensation philosophy before negotiating.",
            "Consider the total package: base salary, variable pay, ESOPs, benefits, and growth opportunities.",
            "Have multiple offers when possible to strengthen your negotiating position.",
            "Be prepared to discuss your contributions and achievements with specific examples and metrics.",
            "Ask about performance review cycles and promotion timelines during negotiation."
        ])
        
        return tips

    def get_benchmark_data(self, role, location, experience, company_size):
        """Get detailed benchmark data"""
        base_data = self.get_market_data(role, location, experience)
        
        # Adjust for company size
        size_multipliers = {
            'startup': 0.9,
            'small': 0.95,
            'medium': 1.0,
            'large': 1.1,
            'enterprise': 1.15
        }
        
        multiplier = size_multipliers.get(company_size.lower(), 1.0)
        
        return