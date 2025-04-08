import requests
import json
import os
from datetime import datetime
from app.models.job import Job
from app.services.nlp import detect_gender_bias, extract_job_keywords
from app import db, app

class JobSearchService:
    """Service for searching and processing job listings."""
    
    def __init__(self):
        self.apis = {
            'indeed': {
                'base_url': 'https://api.indeed.com/ads/apisearch',
                'api_key': app.config['INDEED_API_KEY']
            },
            'linkedin': {
                'base_url': 'https://api.linkedin.com/v2/jobs',
                'api_key': app.config['LINKEDIN_API_KEY']
            },
            'glassdoor': {
                'base_url': 'https://api.glassdoor.com/api/api.htm', 
                'api_key': app.config['GLASSDOOR_API_KEY']
            }
        }
    
    def search_indeed(self, keywords, location, limit=10):
        """Search jobs on Indeed API."""
        try:
            params = {
                'publisher': self.apis['indeed']['api_key'],
                'q': keywords,
                'l': location,
                'limit': limit,
                'format': 'json',
                'v': '2'
            }
            
            response = requests.get(self.apis['indeed']['base_url'], params=params)
            
            if response.status_code == 200:
                data = response.json()
                jobs = []
                
                for job_data in data.get('results', []):
                    # Process job data
                    job = self._process_indeed_job(job_data)
                    jobs.append(job)
                
                return jobs
            else:
                app.logger.error(f"Indeed API error: {response.status_code}")
                return []
        except Exception as e:
            app.logger.error(f"Error searching Indeed jobs: {str(e)}")
            return []
    
    def search_linkedin(self, keywords, location, limit=10):
        """Search jobs on LinkedIn API."""
        try:
            headers = {
                'Authorization': f'Bearer {self.apis["linkedin"]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'keywords': keywords,
                'location': location,
                'count': limit
            }
            
            response = requests.get(self.apis['linkedin']['base_url'], headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                jobs = []
                
                for job_data in data.get('elements', []):
                    # Process job data
                    job = self._process_linkedin_job(job_data)
                    jobs.append(job)
                
                return jobs
            else:
                app.logger.error(f"LinkedIn API error: {response.status_code}")
                return []
        except Exception as e:
            app.logger.error(f"Error searching LinkedIn jobs: {str(e)}")
            return []
    
    def search_jobs(self, keywords, location, sources=None, limit=10):
        """Search jobs across multiple sources."""
        if sources is None:
            sources = ['indeed', 'linkedin']
        
        all_jobs = []
        
        if 'indeed' in sources and self.apis['indeed']['api_key']:
            indeed_jobs = self.search_indeed(keywords, location, limit)
            all_jobs.extend(indeed_jobs)
        
        if 'linkedin' in sources and self.apis['linkedin']['api_key']:
            linkedin_jobs = self.search_linkedin(keywords, location, limit)
            all_jobs.extend(linkedin_jobs)
        
        # Process all jobs for gender bias
        for job in all_jobs:
            self._analyze_job_for_women(job)
        
        return all_jobs
    
    def _process_indeed_job(self, job_data):
        """Process Indeed job data into our Job model."""
        job = Job(
            title=job_data.get('jobtitle', ''),
            company=job_data.get('company', ''),
            location=job_data.get('formattedLocation', ''),
            description=job_data.get('snippet', ''),
            source='indeed',
            external_id=job_data.get('jobkey', ''),
            url=job_data.get('url', ''),
            posted_date=datetime.strptime(job_data.get('date', ''), '%a, %d %b %Y %H:%M:%S %Z') 
                       if job_data.get('date') else None
        )
        return job
    
    def _process_linkedin_job(self, job_data):
        """Process LinkedIn job data into our Job model."""
        job = Job(
            title=job_data.get('title', {}).get('text', ''),
            company=job_data.get('companyDetails', {}).get('name', ''),
            location=job_data.get('locationName', ''),
            description=job_data.get('description', {}).get('text', ''),
            source='linkedin',
            external_id=job_data.get('entityUrn', '').split(':')[-1],
            url=f"https://www.linkedin.com/jobs/view/{job_data.get('entityUrn', '').split(':')[-1]}",
            posted_date=datetime.fromtimestamp(job_data.get('postingDate', 0)/1000) 
                       if job_data.get('postingDate') else None
        )
        return job
    
    def _analyze_job_for_women(self, job):
        """Analyze job listing for women-friendly attributes."""
        if job.description:
            # Detect gender bias in job description
            bias_score = detect_gender_bias(job.description)
            job.bias_score = bias_score
            
            # Check for diversity statements
            diversity_keywords = ['diversity', 'inclusion', 'equal opportunity', 
                                 'women in tech', 'female leadership']
            job.diversity_statement = any(keyword in job.description.lower() for keyword in diversity_keywords)
            
            # Check for parental benefits
            parental_keywords = ['maternity leave', 'paternity leave', 'parental leave',
                               'childcare', 'family-friendly', 'work-life balance']
            job.parental_benefits = any(keyword in job.description.lower() for keyword in parental_keywords)
            
            # Check for flexible hours
            flexibility_keywords = ['flexible hours', 'flexible schedule', 'remote work',
                                  'work from home', 'telecommute']
            job.has_flexible_hours = any(keyword in job.description.lower() for keyword in flexibility_keywords)
            
            # Check for remote work
            remote_keywords = ['remote', 'work from home', 'telecommute', 'virtual position']
            job.is_remote = any(keyword in job.description.lower() for keyword in remote_keywords)
            
            # Process job requirements
            requirements = extract_job_keywords(job.description)
            job.requirements = ', '.join(requirements) if requirements else None
        
        return job
    
    def get_job_recommendations(self, user, limit=10):
        """Get personalized job recommendations for a user."""
        # Create search parameters based on user profile
        keywords = user.desired_job_titles if user.desired_job_titles else ''
        location = user.desired_locations if user.desired_locations else ''
        
        # Get jobs from APIs
        jobs = self.search_jobs(keywords, location, limit=limit)
        
        # Filter jobs based on user preferences
        filtered_jobs = []
        for job in jobs:
            # Check if job meets minimum salary requirement
            if user.min_salary and job.salary_range:
                # Extract numeric salary from range (simplified)
                try:
                    salary_text = job.salary_range.replace('$', '').replace(',', '')
                    salary_parts = salary_text.split('-')
                    min_salary = int(salary_parts[0])
                    if min_salary < user.min_salary:
                        continue
                except:
                    pass  # Skip salary check if parsing fails
            
            # Check remote preference
            if user.remote_preference and not job.is_remote:
                continue
                
            # Check flexible hours preference
            if user.flexible_hours_preference and not job.has_flexible_hours:
                continue
            
            # Check gender bias score
            if job.bias_score and job.bias_score > app.config['BIAS_THRESHOLD']:
                continue
                
            filtered_jobs.append(job)
        
        return filtered_jobs