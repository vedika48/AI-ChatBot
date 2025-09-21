"""
Real-time Job Matching System with Semantic Matching
Author: AI Assistant
Description: Fetches and matches jobs from multiple APIs using async calls and semantic matching
"""

import os
import requests
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Apply nest_asyncio for better async handling in Jupyter environments
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobAPIClient:
    """Async client for fetching real-time job data from various APIs"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        # Load API keys from environment variables if not provided
        if api_keys is None:
            api_keys = {
                'adzuna_app_id': os.getenv('ADZUNA_APP_ID'),
                'adzuna_app_key': os.getenv('ADZUNA_APP_KEY'),
                'rapidapi_key': os.getenv('RAPIDAPI_KEY'),
                'jooble_api_key': os.getenv('JOOBLE_API_KEY')
            }
        
        self.api_keys = api_keys
        self.headers = {
            'User-Agent': 'JobMatcher/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.request_delay = 0.5  # Reduced delay for async calls
        self.timeout = aiohttp.ClientTimeout(total=20)
    
    async def _make_async_request(self, session, url, method='GET', params=None, json_data=None, headers=None):
        """Generic async request method"""
        try:
            if method.upper() == 'GET':
                async with session.get(url, params=params, headers=headers, timeout=self.timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.upper() == 'POST':
                async with session.post(url, json=json_data, headers=headers, timeout=self.timeout) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"Async request error: {e}")
            return None
    
    async def fetch_jobs_adzuna(self, keywords: str, location: str, country: str = 'in', 
                               results_per_page: int = 50) -> List[Dict]:
        """Fetch jobs from Adzuna API asynchronously"""
        app_id = self.api_keys.get('adzuna_app_id')
        app_key = self.api_keys.get('adzuna_app_key')
        
        if not app_id or not app_key:
            logger.warning("Adzuna API credentials not provided")
            return []
            
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            'app_id': app_id,
            'app_key': app_key,
            'what': keywords,
            'where': location,
            'results_per_page': min(results_per_page, 50),
            'sort_by': 'relevance'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(self.request_delay)
                data = await self._make_async_request(session, url, params=params, headers=self.headers)
                
                if not data:
                    return []
                
                jobs = []
                for job in data.get('results', []):
                    salary_min = job.get('salary_min')
                    salary_max = job.get('salary_max')
                    salary_range = None
                    
                    if salary_min and salary_max:
                        salary_range = f"₹{salary_min:,.0f} - ₹{salary_max:,.0f}"
                    elif salary_min:
                        salary_range = f"₹{salary_min:,.0f}+"
                    
                    jobs.append({
                        'job_id': job.get('id', ''),
                        'job_title': job.get('title', ''),
                        'company': job.get('company', {}).get('display_name', ''),
                        'location': job.get('location', {}).get('display_name', ''),
                        'job_description': job.get('description', ''),
                        'salary_min': salary_min,
                        'salary_max': salary_max,
                        'salary_range': salary_range,
                        'job_url': job.get('redirect_url', ''),
                        'created_date': job.get('created', ''),
                        'contract_type': job.get('contract_type', ''),
                        'category': job.get('category', {}).get('label', ''),
                        'source': 'adzuna'
                    })
                
                logger.info(f"Fetched {len(jobs)} jobs from Adzuna")
                return jobs
                
        except Exception as e:
            logger.error(f"Error in Adzuna async fetch: {e}")
            return []
    
    async def fetch_jobs_jsearch(self, keywords: str, location: str, limit: int = 25) -> List[Dict]:
        """Fetch jobs from JSearch API via RapidAPI - More reliable than Indeed"""
        api_key = self.api_keys.get('rapidapi_key')
        
        if not api_key:
            logger.warning("RapidAPI key not provided for JSearch")
            return []
            
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            **self.headers,
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
        }
        
        params = {
            'query': f'{keywords} in {location}',
            'page': '1',
            'num_pages': '1',
            'date_posted': 'month'  # Get recent jobs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(self.request_delay)
                data = await self._make_async_request(session, url, params=params, headers=headers)
                
                if not data or 'data' not in data:
                    logger.warning("No data received from JSearch API")
                    return []
                
                jobs = []
                for job in data.get('data', [])[:limit]:
                    # Extract salary information
                    salary_min = job.get('job_min_salary')
                    salary_max = job.get('job_max_salary')
                    salary_range = None
                    
                    if salary_min and salary_max:
                        salary_range = f"₹{salary_min:,.0f} - ₹{salary_max:,.0f}"
                    elif salary_min:
                        salary_range = f"₹{salary_min:,.0f}+"
                    
                    jobs.append({
                        'job_id': job.get('job_id', ''),
                        'job_title': job.get('job_title', ''),
                        'company': job.get('employer_name', ''),
                        'location': job.get('job_location', ''),
                        'job_description': job.get('job_description', ''),
                        'salary_min': salary_min,
                        'salary_max': salary_max,
                        'salary_range': salary_range,
                        'job_url': job.get('job_apply_link', ''),
                        'created_date': job.get('job_posted_at_datetime_utc', ''),
                        'contract_type': job.get('job_employment_type', ''),
                        'category': job.get('job_job_title', ''),
                        'source': 'jsearch'
                    })
                
                logger.info(f"Fetched {len(jobs)} jobs from JSearch")
                return jobs
                
        except Exception as e:
            logger.error(f"Error in JSearch async fetch: {e}")
            return []
    
    async def fetch_jobs_jooble(self, keywords: str, location: str, limit: int = 20) -> List[Dict]:
        """Fetch jobs from Jooble API asynchronously"""
        api_key = self.api_keys.get('jooble_api_key')
        
        if not api_key:
            logger.warning("Jooble API key not provided")
            return []
            
        url = f"https://jooble.org/api/{api_key}"
        payload = {
            'keywords': keywords,
            'location': location,
            'searchMode': '1',
            'page': '1'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(self.request_delay)
                data = await self._make_async_request(session, url, method='POST', json_data=payload, headers=self.headers)
                
                if not data:
                    return []
                
                jobs = []
                for job in data.get('jobs', [])[:limit]:
                    jobs.append({
                        'job_id': job.get('id', ''),
                        'job_title': job.get('title', ''),
                        'company': job.get('company', ''),
                        'location': job.get('location', ''),
                        'job_description': job.get('snippet', ''),
                        'salary_min': job.get('salary'),
                        'salary_max': None,
                        'salary_range': job.get('salary'),
                        'job_url': job.get('link', ''),
                        'created_date': job.get('updated', ''),
                        'contract_type': job.get('type', ''),
                        'category': '',
                        'source': 'jooble'
                    })
                
                logger.info(f"Fetched {len(jobs)} jobs from Jooble")
                return jobs
                
        except Exception as e:
            logger.error(f"Error in Jooble async fetch: {e}")
            return []
    
    async def fetch_all_jobs_async(self, keywords: str, location: str) -> List[Dict]:
        """Fetch jobs from all APIs concurrently"""
        tasks = [
            self.fetch_jobs_adzuna(keywords, location),
            self.fetch_jobs_jsearch(keywords, location),  # Using JSearch instead of Indeed
            self.fetch_jobs_jooble(keywords, location)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_jobs = []
            for result in results:
                if isinstance(result, list):
                    all_jobs.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"API fetch error: {result}")
            
            return all_jobs
            
        except Exception as e:
            logger.error(f"Error in concurrent fetch: {e}")
            return []

class RealTimeJobMatcher:
    """Main job matching class with semantic matching using sentence-transformers"""
    
    def __init__(self, api_keys: Dict[str, str] = None, model_name: str = 'all-MiniLM-L6-v2'):
        self.api_client = JobAPIClient(api_keys)
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to smaller model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded fallback model: all-MiniLM-L6-v2")
            except:
                logger.error("Failed to load any sentence transformer model")
                self.model = None
        
        self.job_cache = {}
        self.cache_expiry = timedelta(hours=1)
        self.embedding_cache = {}  # Cache for job embeddings
    
    async def fetch_and_cache_jobs_async(self, keywords: str, location: str, 
                                       refresh_cache: bool = False) -> List[Dict]:
        """Fetch jobs from multiple APIs asynchronously and cache them"""
        cache_key = f"{keywords.lower()}_{location.lower()}"
        current_time = datetime.now()
        
        # Check cache
        if not refresh_cache and cache_key in self.job_cache:
            cached_data, timestamp = self.job_cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                logger.info("Using cached job data")
                return cached_data
        
        logger.info(f"Fetching fresh job data for '{keywords}' in '{location}'")
        
        # Fetch jobs asynchronously
        all_jobs = await self.api_client.fetch_all_jobs_async(keywords, location)
        
        # Remove duplicates
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            job_key = (
                job['job_title'].lower().strip(),
                job['company'].lower().strip(),
                job['location'].lower().strip()
            )
            
            if job_key not in seen and job['job_title'].strip():
                seen.add(job_key)
                unique_jobs.append(job)
        
        # Cache results
        if unique_jobs:
            self.job_cache[cache_key] = (unique_jobs, current_time)
        
        logger.info(f"Fetched {len(unique_jobs)} unique jobs from {len(all_jobs)} total")
        return unique_jobs
    
    def _get_job_embedding(self, job_text: str) -> np.ndarray:
        """Get embedding for job text with caching"""
        if job_text in self.embedding_cache:
            return self.embedding_cache[job_text]
        
        if self.model:
            embedding = self.model.encode(job_text, convert_to_tensor=False)
            self.embedding_cache[job_text] = embedding
            return embedding
        else:
            # Fallback: return zero vector if model not available
            return np.zeros(384)  # Default size for MiniLM models
    
    def calculate_semantic_match_score(self, user_profile: Dict, job: Dict) -> Tuple[float, Dict]:
        """Calculate semantic match score using sentence transformers"""
        if not self.model:
            return 0.0, {'semantic_match': 0.0}
        
        # Prepare text for embedding
        user_skills = user_profile.get('skills', '')
        user_preferences = user_profile.get('preferred_roles', '')
        user_text = f"{user_skills} {user_preferences}".strip()
        
        job_title = job.get('job_title', '')
        job_description = job.get('job_description', '')
        job_text = f"{job_title} {job_description}".strip()
        
        if not user_text or not job_text:
            return 0.0, {'semantic_match': 0.0}
        
        try:
            # Get embeddings
            user_embedding = self._get_job_embedding(user_text)
            job_embedding = self._get_job_embedding(job_text)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(
                user_embedding.reshape(1, -1), 
                job_embedding.reshape(1, -1)
            ).item()
            
            # Normalize to 0-1 range
            semantic_score = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return semantic_score, {'semantic_match': semantic_score}
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return 0.0, {'semantic_match': 0.0}
    
    def calculate_job_match_score(self, user_profile: Dict, job: Dict) -> Tuple[float, Dict]:
        """Calculate comprehensive match score with semantic matching"""
        score_components = {}
        
        # Semantic matching (50% weight)
        semantic_score, semantic_components = self.calculate_semantic_match_score(user_profile, job)
        score_components.update(semantic_components)
        score_components['semantic_match_weighted'] = semantic_score * 0.5
        
        # Location matching (20% weight)
        user_location = user_profile.get('location', '').lower()
        job_location = job.get('location', '').lower()
        location_match = 0.2 if user_location and job_location and (
            user_location in job_location or job_location in user_location
        ) else 0
        score_components['location_match'] = location_match
        
        # Experience matching (15% weight)
        user_exp = user_profile.get('experience_years', 0)
        job_text = f"{job['job_title']} {job['job_description']}".lower()
        experience_bonus = 0
        
        if user_exp >= 0:
            if any(word in job_text for word in ['senior', 'lead', 'principal', 'manager']):
                required_exp = 5
            elif any(word in job_text for word in ['mid', 'intermediate', 'experienced']):
                required_exp = 3
            elif any(word in job_text for word in ['junior', 'entry', 'fresher', 'graduate']):
                required_exp = 1
            else:
                required_exp = 2
            
            exp_diff = abs(user_exp - required_exp)
            if exp_diff <= 1:
                experience_bonus = 0.15
            elif exp_diff <= 2:
                experience_bonus = 0.10
            elif exp_diff <= 3:
                experience_bonus = 0.05
        
        score_components['experience_match'] = experience_bonus
        
        # Salary expectation matching (15% weight)
        user_expected_salary = user_profile.get('expected_salary', 0)
        job_salary_min = job.get('salary_min')
        
        salary_match = 0
        if user_expected_salary > 0 and job_salary_min and job_salary_min > 0:
            salary_ratio = min(user_expected_salary, job_salary_min) / max(user_expected_salary, job_salary_min)
            salary_match = salary_ratio * 0.15
        
        score_components['salary_match'] = salary_match
        
        # Calculate total score
        total_score = sum([
            score_components['semantic_match_weighted'],
            score_components['location_match'],
            score_components['experience_match'],
            score_components['salary_match']
        ])
        
        return total_score, score_components
    
    async def recommend_jobs_async(self, user_profile: Dict, top_k: int = 10, 
                                 refresh_cache: bool = False) -> List[Dict]:
        """Async method to recommend jobs based on user profile"""
        user_skills = user_profile.get('skills', '')
        user_location = user_profile.get('location', 'bangalore')
        preferred_roles = user_profile.get('preferred_roles', user_skills)
        
        search_terms = preferred_roles if preferred_roles else user_skills
        
        # Fetch jobs asynchronously
        jobs = await self.fetch_and_cache_jobs_async(search_terms, user_location, refresh_cache)
        
        if not jobs:
            logger.warning("No jobs found")
            return []
        
        df = pd.DataFrame(jobs)
        df['job_description'] = df['job_description'].fillna('')
        
        # Calculate match scores
        job_scores = []
        for idx, job in df.iterrows():
            job_dict = job.to_dict()
            match_score, score_components = self.calculate_job_match_score(user_profile, job_dict)
            job_scores.append({
                'index': idx,
                'score': match_score,
                'components': score_components
            })
        
        # Sort and get top recommendations
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        top_jobs = job_scores[:top_k]
        
        recommendations = []
        for job_score in top_jobs:
            idx = job_score['index']
            job = df.iloc[idx]
            
            description = job['job_description']
            if len(description) > 300:
                description = description[:300] + '...'
            
            rec = {
                'job_id': job.get('job_id', ''),
                'job_title': job['job_title'],
                'company': job['company'],
                'location': job['location'],
                'job_description': description,
                'salary_min': job.get('salary_min'),
                'salary_max': job.get('salary_max'),
                'salary_range': job.get('salary_range'),
                'match_score': round(job_score['score'], 3),
                'match_details': job_score['components'],
                'job_url': job.get('job_url', ''),
                'created_date': job.get('created_date', ''),
                'contract_type': job.get('contract_type', ''),
                'category': job.get('category', ''),
                'source': job.get('source', '')
            }
            recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} job recommendations")
        return recommendations
    
    async def get_job_statistics_async(self, keywords: str, location: str) -> Dict[str, Any]:
        """Async method to get job market statistics"""
        jobs = await self.fetch_and_cache_jobs_async(keywords, location)
        
        if not jobs:
            return {}
        
        df = pd.DataFrame(jobs)
        
        stats = {
            'total_jobs': len(jobs),
            'unique_companies': df['company'].nunique(),
            'top_companies': df['company'].value_counts().head(5).to_dict(),
            'locations': df['location'].value_counts().to_dict(),
            'contract_types': df['contract_type'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'avg_jobs_per_company': round(len(jobs) / max(1, df['company'].nunique()), 2),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Salary statistics
        salary_jobs = df[df['salary_min'].notna()]
        if not salary_jobs.empty:
            stats['salary_stats'] = {
                'jobs_with_salary': len(salary_jobs),
                'avg_min_salary': round(salary_jobs['salary_min'].mean(), 2),
                'median_min_salary': round(salary_jobs['salary_min'].median(), 2),
                'salary_range': {
                    'min': salary_jobs['salary_min'].min(),
                    'max': salary_jobs['salary_max'].max() if 'salary_max' in salary_jobs.columns else None
                }
            }
        
        return stats

class SyncJobFetcher:
    """Synchronous wrapper for the async job matcher"""
    
    def __init__(self, api_keys: Dict[str, str] = None, model_name: str = 'all-MiniLM-L6-v2'):
        self.async_matcher = RealTimeJobMatcher(api_keys, model_name)
    
    def fetch_jobs_sync(self, keywords: str, location: str, limit: int = 20):
        """Synchronous method to fetch jobs"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            jobs = loop.run_until_complete(
                self.async_matcher.fetch_and_cache_jobs_async(keywords, location)
            )
            
            loop.close()
            return jobs[:limit] if jobs else []
            
        except Exception as e:
            logger.error(f"Sync fetch error: {e}")
            return []
    
    def recommend_jobs_sync(self, user_profile: Dict, top_k: int = 10):
        """Synchronous method to get job recommendations"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            recommendations = loop.run_until_complete(
                self.async_matcher.recommend_jobs_async(user_profile, top_k)
            )
            
            loop.close()
            return recommendations
            
        except Exception as e:
            logger.error(f"Sync recommendations error: {e}")
            return []