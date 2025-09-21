import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta

class JobScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.popular_locations = [
            'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 
            'Pune', 'Kolkata', 'Ahmedabad', 'Kochi', 'Gurgaon'
        ]
        self.trending_skills = [
            'Python', 'Java', 'JavaScript', 'React', 'Node.js', 'AWS', 
            'Machine Learning', 'Data Science', 'DevOps', 'Docker', 
            'Kubernetes', 'Angular', 'Spring Boot', 'MongoDB'
        ]

    def get_latest_jobs(self, skills='', location='bangalore', experience='mid'):
        """Get latest job postings from multiple sources"""
        jobs = []
        
        # Simulate job data (in production, integrate with actual APIs)
        sample_jobs = [
            {
                'title': 'Senior Software Engineer - Python',
                'company': 'Flipkart',
                'location': 'Bangalore',
                'experience': '3-5 years',
                'salary_range': '15-25 LPA',
                'skills_required': ['Python', 'Django', 'REST APIs', 'PostgreSQL'],
                'job_url': 'https://www.flipkartcareers.com/job123',
                'posted_date': '2024-01-15',
                'job_type': 'Full-time',
                'remote_option': False,
                'women_friendly_score': 4.5
            },
            {
                'title': 'Frontend Developer - React',
                'company': 'Swiggy',
                'location': 'Bangalore',
                'experience': '2-4 years',
                'salary_range': '12-20 LPA',
                'skills_required': ['React', 'JavaScript', 'HTML/CSS', 'Redux'],
                'job_url': 'https://careers.swiggy.com/job456',
                'posted_date': '2024-01-14',
                'job_type': 'Full-time',
                'remote_option': True,
                'women_friendly_score': 4.3
            },
            {
                'title': 'Data Scientist',
                'company': 'Zomato',
                'location': 'Delhi',
                'experience': '2-5 years',
                'salary_range': '18-28 LPA',
                'skills_required': ['Python', 'Machine Learning', 'SQL', 'Tableau'],
                'job_url': 'https://www.zomato.com/careers/job789',
                'posted_date': '2024-01-13',
                'job_type': 'Full-time',
                'remote_option': True,
                'women_friendly_score': 4.2
            }
        ]
        
        return sample_jobs

    def search_jobs(self, query='', location='', company='', experience='', salary_min=0, remote=False):
        """Search jobs with specific criteria"""
        # In production, this would integrate with job portal APIs
        # For now, return filtered sample data
        
        all_jobs = self.get_latest_jobs()
        filtered_jobs = []
        
        for job in all_jobs:
            match = True
            
            if query and query.lower() not in job['title'].lower():
                match = False
            if location and location.lower() not in job['location'].lower():
                match = False
            if company and company.lower() not in job['company'].lower():
                match = False
            if remote and not job['remote_option']:
                match = False
                
            if match:
                filtered_jobs.append(job)
                
        return filtered_jobs

    def get_popular_locations(self):
        return self.popular_locations

    def get_trending_skills(self):
        return self.trending_skills

    def get_top_companies(self):
        return [
            'Flipkart', 'Amazon', 'Google', 'Microsoft', 'Swiggy', 
            'Zomato', 'Paytm', 'Ola', 'PhonePe', 'Razorpay'
        ]