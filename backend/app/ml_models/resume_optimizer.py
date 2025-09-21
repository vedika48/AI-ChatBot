"""
AI-Powered Resume Optimizer
Enhanced version with better error handling and performance
"""

import re
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import json
from datetime import datetime
import io

# Handle optional dependencies
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    """Parse and extract information from resume documents"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced industry-specific skill keywords
        self.skill_keywords = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
                'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'nosql', 'graphql'
            ],
            'web_development': [
                'react', 'angular', 'vue', 'svelte', 'nodejs', 'express', 'django', 'flask',
                'fastapi', 'html', 'css', 'bootstrap', 'tailwind', 'sass', 'webpack', 'vite', 'npm', 'yarn'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'ai', 'artificial intelligence',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda',
                'statistical analysis', 'data visualization', 'big data', 'hadoop', 'spark', 'pyspark'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git',
                'github', 'gitlab', 'terraform', 'ansible', 'ci/cd', 'devops', 'microservices',
                'linux', 'bash', 'shell scripting', 'nginx', 'apache'
            ],
            'databases': [
                'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
                'oracle', 'sqlite', 'cassandra', 'dynamodb', 'cosmosdb', 'snowflake'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving',
                'project management', 'analytical thinking', 'creativity', 'adaptability',
                'time management', 'critical thinking', 'decision making', 'negotiation'
            ]
        }
        
    def extract_text_from_pdf(self, file_content: bytes) -> Optional[str]:
        """Extract text from PDF file with better error handling"""
        if not HAS_PYPDF2:
            logger.error("PyPDF2 not installed. Cannot parse PDF files.")
            return None
            
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def extract_text_from_docx(self, file_content: bytes) -> Optional[str]:
        """Extract text from DOCX file with better error handling"""
        if not HAS_DOCX:
            logger.error("python-docx not installed. Cannot parse DOCX files.")
            return None
            
        try:
            doc_file = io.BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return None
    
    def extract_text_from_file(self, file_content: bytes, file_type: str) -> Optional[str]:
        """Extract text from various file types"""
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file_content)
        elif file_type == 'txt':
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return file_content.decode('latin-1')
                except Exception as e:
                    logger.error(f"Error decoding text file: {e}")
                    return None
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Enhanced contact information extraction"""
        contact_info = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': '',
            'portfolio': ''
        }
        
        if not text:
            return contact_info
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone extraction (improved patterns)
        phone_patterns = [
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?91[-.\s]?\d{5}[-.\s]?\d{5}',
            r'\+?91[-.\s]?\d{10}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{5}[-.\s]?\d{5}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info['phone'] = phones[0]
                break
        
        # LinkedIn extraction
        linkedin_patterns = [
            r'linkedin\.com/in/[\w-]+',
            r'linkedin\.com/company/[\w-]+',
            r'linkedin\.com/profile/view\?id=\d+'
        ]
        
        for pattern in linkedin_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                contact_info['linkedin'] = f"https://{matches[0]}"
                break
        
        # GitHub extraction
        github_pattern = r'github\.com/[\w-]+'
        github_matches = re.findall(github_pattern, text.lower())
        if github_matches:
            contact_info['github'] = f"https://{github_matches[0]}"
        
        # Portfolio website extraction
        portfolio_patterns = [
            r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|org|net|io|dev)\b',
            r'\b(?:portfolio|website):\s*(https?://[^\s]+)'
        ]
        
        for pattern in portfolio_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                contact_info['portfolio'] = matches[0] if matches[0].startswith('http') else f'https://{matches[0]}'
                break
        
        return contact_info
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Enhanced skills extraction with better matching"""
        if not text:
            return {category: [] for category in self.skill_keywords}
            
        text_lower = text.lower()
        found_skills = {category: [] for category in self.skill_keywords}
        
        for category, skills in self.skill_keywords.items():
            for skill in skills:
                # More flexible pattern matching
                skill_patterns = [
                    rf'\b{re.escape(skill)}\b',
                    rf'\b{re.escape(skill.replace(" ", "-"))}\b',
                    rf'\b{re.escape(skill.replace(" ", "_"))}\b',
                    rf'\b{re.escape(skill.replace(" ", ""))}\b'
                ]
                
                for pattern in skill_patterns:
                    if re.search(pattern, text_lower):
                        if skill not in found_skills[category]:  # Avoid duplicates
                            found_skills[category].append(skill)
                        break
        
        return found_skills

    def extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience from resume text"""
        experience = []
        
        # Look for common experience section headers
        experience_sections = re.split(
            r'\n(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY)\n',
            text, flags=re.IGNORECASE
        )
        
        if len(experience_sections) > 1:
            exp_text = experience_sections[1]
            
            # Split by common job entry patterns
            job_entries = re.split(r'\n(?=[A-Z][a-zA-Z\s]+(?:Engineer|Manager|Developer|Analyst|Specialist|Consultant))', exp_text)
            
            for entry in job_entries[:5]:  # Limit to first 5 entries
                if len(entry.strip()) > 50:  # Filter out short entries
                    # Extract job title, company, dates
                    lines = [line.strip() for line in entry.split('\n') if line.strip()]
                    if lines:
                        job_info = {
                            'title': lines[0] if lines else '',
                            'company': '',
                            'duration': '',
                            'description': entry.strip()
                        }
                        
                        # Try to extract company and dates from subsequent lines
                        for line in lines[1:3]:
                            # Look for date patterns
                            date_pattern = r'(20\d{2}|19\d{2})|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                            if re.search(date_pattern, line):
                                job_info['duration'] = line
                            else:
                                job_info['company'] = line
                        
                        experience.append(job_info)
        
        return experience
    
    def calculate_experience_years(self, text: str) -> float:
        """Calculate total years of experience from resume"""
        # Look for explicit experience mentions
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'experience\s*of\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return float(matches[0])
        
        # Try to calculate from work experience dates
        date_pattern = r'(20\d{2}|19\d{2})'
        years = re.findall(date_pattern, text)
        
        if len(years) >= 2:
            years = [int(year) for year in years]
            return max(years) - min(years)
        
        return 0.0
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from resume"""
        education = []
        
        # Look for education section
        education_sections = re.split(
            r'\n(?:EDUCATION|ACADEMIC BACKGROUND|QUALIFICATIONS)\n',
            text, flags=re.IGNORECASE
        )
        
        if len(education_sections) > 1:
            edu_text = education_sections[1]
            
            # Common degree patterns
            degree_patterns = [
                r'(Bachelor|B\.?Tech|B\.?E\.?|B\.?S\.?|B\.?A\.?)\s*(?:of|in)?\s*([A-Za-z\s]+)',
                r'(Master|M\.?Tech|M\.?S\.?|M\.?A\.?|MBA)\s*(?:of|in)?\s*([A-Za-z\s]+)',
                r'(PhD|Ph\.?D\.?|Doctorate)\s*(?:of|in)?\s*([A-Za-z\s]+)'
            ]
            
            for pattern in degree_patterns:
                matches = re.findall(pattern, edu_text, re.IGNORECASE)
                for match in matches:
                    education.append({
                        'degree': match[0],
                        'field': match[1].strip(),
                        'institution': '',
                        'year': ''
                    })
        
        return education

class ResumeOptimizer:
    """Enhanced Resume Optimizer with better performance and error handling"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.parser = ResumeParser()
        self.api_keys = api_keys or {}
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._analysis_cache = {}  # Cache resume analysis results
        
    def analyze_resume(self, resume_text: str, use_cache: bool = True) -> Dict[str, Any]:
        """Cached resume analysis for better performance"""
        if not resume_text or not isinstance(resume_text, str):
            return self._get_empty_analysis()
            
        # Use cache for performance
        cache_key = hash(resume_text)
        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        try:
            analysis = {
                'contact_info': self.parser.extract_contact_info(resume_text),
                'skills': self.parser.extract_skills(resume_text),
                'experience': self.parser.extract_experience(resume_text),
                'experience_years': self.parser.calculate_experience_years(resume_text),
                'education': self.parser.extract_education(resume_text),
                'resume_length': len(resume_text),
                'word_count': len(resume_text.split()),
                'readability_score': self._calculate_readability(resume_text),
                'keyword_density': self._calculate_keyword_density(resume_text),
                'ats_score': self._calculate_ats_score(resume_text),
                'resume_text': resume_text  # Store for later use
            }
            
            # Cache the result
            self._analysis_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            return self._get_empty_analysis()
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'contact_info': {'email': '', 'phone': '', 'linkedin': '', 'github': '', 'portfolio': ''},
            'skills': {category: [] for category in self.parser.skill_keywords},
            'experience': [],
            'experience_years': 0.0,
            'education': [],
            'resume_length': 0,
            'word_count': 0,
            'readability_score': 0.0,
            'keyword_density': {},
            'ats_score': 0.0,
            'resume_text': ''
        }

    def optimize_for_job(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Optimize resume for a specific job description with better error handling"""
        if not resume_text or not job_description:
            return self._get_empty_optimization()
            
        try:
            # Analyze current resume (cached)
            resume_analysis = self.analyze_resume(resume_text)
            
            # Analyze job description
            job_keywords = self._extract_job_keywords(job_description)
            required_skills = self._extract_required_skills(job_description)
            
            # Calculate match score
            match_score = self._calculate_job_match_score(resume_text, job_description)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                resume_analysis, job_keywords, required_skills, job_description
            )
            
            # Generate optimized resume sections
            optimized_sections = self._generate_optimized_sections(
                resume_text, job_description, suggestions
            )
            
            return {
                'current_analysis': resume_analysis,
                'job_keywords': job_keywords,
                'required_skills': required_skills,
                'match_score': match_score,
                'suggestions': suggestions,
                'optimized_sections': optimized_sections,
                'missing_skills': self._find_missing_skills(resume_analysis['skills'], required_skills),
                'keyword_recommendations': self._get_keyword_recommendations(resume_text, job_description)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing resume for job: {e}")
            return self._get_empty_optimization()
    
    def _get_empty_optimization(self) -> Dict[str, Any]:
        """Return empty optimization structure"""
        return {
            'current_analysis': self._get_empty_analysis(),
            'job_keywords': [],
            'required_skills': {category: [] for category in self.parser.skill_keywords},
            'match_score': 0.0,
            'suggestions': [],
            'optimized_sections': {},
            'missing_skills': {},
            'keyword_recommendations': {'missing_keywords': [], 'keyword_integration_tips': []}
        }

    # Enhanced version of _generate_optimized_summary
    def _generate_optimized_summary(self, resume_text: str, job_description: str, 
                                  job_keywords: List[str]) -> str:
        """Generate optimized professional summary with better error handling"""
        try:
            resume_analysis = self.analyze_resume(resume_text)
            experience_years = resume_analysis['experience_years']
            
            # Handle case with no experience
            if experience_years <= 0:
                experience_phrase = "Entry-level professional"
            else:
                experience_phrase = f"Results-driven professional with {experience_years:.0f} years of experience"
            
            # Extract top skills
            all_skills = []
            for skill_list in resume_analysis['skills'].values():
                all_skills.extend(skill_list)
            
            top_skills = all_skills[:3] if all_skills else ['technical', 'problem-solving']
            
            # Include relevant job keywords
            relevant_keywords = [kw for kw in job_keywords[:3] if len(kw) > 4]
            if not relevant_keywords:
                relevant_keywords = ['technology', 'development', 'solutions']
            
            summary = f"{experience_phrase} in {', '.join(top_skills)}. "
            summary += f"Proven expertise in {', '.join(relevant_keywords)} with a track record of "
            summary += "delivering high-quality solutions and driving business growth. "
            summary += "Passionate about leveraging technology to solve complex problems and "
            summary += "contribute to organizational success."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating optimized summary: {e}")
            return "Experienced professional with strong technical skills and proven track record of delivering results."