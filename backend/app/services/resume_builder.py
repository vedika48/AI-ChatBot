import re
from datetime import datetime
import json
from typing import Dict, List

class ResumeBuilder:
    def __init__(self):
        self.indian_resume_keywords = {
            'technical_skills': [
                'python', 'java', 'javascript', 'react', 'angular', 'node.js',
                'spring boot', 'hibernate', 'mysql', 'mongodb', 'redis',
                'aws', 'docker', 'kubernetes', 'jenkins', 'git'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical thinking', 'project management', 'agile', 'scrum'
            ],
            'indian_context': [
                'client interaction', 'onsite opportunity', 'offshore development',
                'cross-cultural communication', 'multi-time zone coordination'
            ]
        }
    
    def validate_user_profile(self, user_profile: Dict) -> List[str]:
        """Validate user profile data and return errors"""
        errors = []
        
        required_fields = ['name', 'email', 'skills']
        for field in required_fields:
            if not user_profile.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate email format
        if user_profile.get('email') and not re.match(r'^[^@]+@[^@]+\.[^@]+$', user_profile['email']):
            errors.append("Invalid email format")
        
        return errors

    def generate_resume(self, user_profile, template='professional', target_role=''):
        """Generate resume content based on user profile"""
        resume_content = {
            'personal_info': self._generate_personal_info(user_profile),
            'professional_summary': self._generate_summary(user_profile, target_role),
            'technical_skills': self._organize_skills(user_profile.get('skills', [])),
            'work_experience': self._format_experience(user_profile.get('experience', [])),
            'education': self._format_education(user_profile.get('education', {})),
            'projects': self._format_projects(user_profile.get('projects', [])),
            'certifications': user_profile.get('certifications', []),
            'achievements': user_profile.get('achievements', [])
        }
        
        return resume_content

    def calculate_ats_score(self, resume_text):
        """Calculate ATS compatibility score"""
        score = 0
        max_score = 100
        
        # Check for standard sections
        sections = ['experience', 'education', 'skills', 'summary']
        section_score = 0
        for section in sections:
            if section.lower() in resume_text.lower():
                section_score += 20
        score += min(section_score, 60)  # Max 60 points for sections
        
        # Check for quantified achievements
        if re.search(r'\d+%|\d+\s*(years?|months?)', resume_text):
            score += 15
        
        # Check for relevant keywords
        keyword_count = sum(1 for keyword in self.indian_resume_keywords['technical_skills'] 
                           if keyword in resume_text.lower())
        score += min(keyword_count * 2, 20)  # Max 20 points for keywords
        
        # Check for proper formatting (simple heuristics)
        if len(resume_text.split('\n')) > 10:  # Has line breaks
            score += 5
        
        return min(score, max_score)

    def analyze_keyword_density(self, resume_text, job_description):
        """Analyze keyword match between resume and job description"""
        if not job_description:
            return {'match_percentage': 0, 'matched_keywords': [], 'missing_keywords': []}
        
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        # Focus on technical keywords
        all_keywords = (self.indian_resume_keywords['technical_skills'] + 
                       self.indian_resume_keywords['soft_skills'])
        
        job_keywords = [word for word in all_keywords if word in job_description.lower()]
        matched_keywords = [word for word in job_keywords if word in resume_text.lower()]
        missing_keywords = [word for word in job_keywords if word not in resume_text.lower()]
        
        match_percentage = (len(matched_keywords) / len(job_keywords) * 100) if job_keywords else 0
        
        return {
            'match_percentage': round(match_percentage, 1),
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords[:10],  # Top 10 missing keywords
            'total_job_keywords': len(job_keywords)
        }

    def analyze_sections(self, resume_text):
        """Analyze resume sections and provide feedback"""
        sections_analysis = {}
        
        # Professional Summary Analysis
        if 'summary' in resume_text.lower() or 'objective' in resume_text.lower():
            sections_analysis['summary'] = {
                'present': True,
                'feedback': 'Great! Professional summary found.',
                'suggestions': ['Keep it concise (2-3 lines)', 'Include your years of experience', 'Mention key skills']
            }
        else:
            sections_analysis['summary'] = {
                'present': False,
                'feedback': 'Missing professional summary',
                'suggestions': ['Add a 2-3 line professional summary at the top', 'Highlight your experience and key strengths']
            }
        
        # Skills Section Analysis
        if 'skills' in resume_text.lower() or 'technical' in resume_text.lower():
            sections_analysis['skills'] = {
                'present': True,
                'feedback': 'Skills section found',
                'suggestions': ['Group skills by category (Programming Languages, Frameworks, Tools)', 'Include both technical and soft skills']
            }
        else:
            sections_analysis['skills'] = {
                'present': False,
                'feedback': 'Missing skills section',
                'suggestions': ['Add a dedicated skills section', 'Include relevant technical skills for your target role']
            }
        
        # Experience Section Analysis
        if 'experience' in resume_text.lower() or 'work' in resume_text.lower():
            sections_analysis['experience'] = {
                'present': True,
                'feedback': 'Work experience section found',
                'suggestions': ['Use bullet points for achievements', 'Quantify your impact with numbers', 'Include relevant technologies used']
            }
        else:
            sections_analysis['experience'] = {
                'present': False,
                'feedback': 'Missing work experience section',
                'suggestions': ['Add work experience with company names and dates', 'Focus on achievements rather than just responsibilities']
            }
        
        return sections_analysis

    def get_formatting_tips(self):
        """Get formatting tips for Indian resumes"""
        return [
            "Keep your resume to 1-2 pages maximum",
            "Use a clean, professional font like Arial or Calibri (10-12pt)",
            "Include your professional photo (common practice in India)",
            "Add your current CTC and expected CTC",
            "Mention your notice period (usually 30-90 days)",
            "Include your LinkedIn profile URL",
            "Use consistent formatting for dates (MM/YYYY)",
            "Save in both PDF and Word formats",
            "Ensure proper spacing and margins for readability",
            "Use bullet points for easy scanning"
        ]

    def get_optimization_tips(self, user_profile, target_role):
        """Get role-specific optimization tips"""
        tips = []
        
        role = target_role.lower()
        
        if 'software' in role or 'developer' in role:
            tips.extend([
                "Highlight programming languages and frameworks you've used",
                "Include links to your GitHub profile and portfolio",
                "Mention specific projects with technologies used",
                "Add any open-source contributions",
                "Include relevant certifications (AWS, Google Cloud, etc.)"
            ])
        
        elif 'data' in role:
            tips.extend([
                "Emphasize statistical analysis and machine learning skills",
                "Include experience with data visualization tools",
                "Mention big data technologies and databases",
                "Highlight any research or publication experience",
                "Include relevant Python/R libraries you've used"
            ])
        
        elif 'product' in role:
            tips.extend([
                "Highlight user research and market analysis experience",
                "Include metrics showing product impact",
                "Mention agile/scrum methodology experience",
                "Emphasize cross-functional collaboration skills",
                "Include any product management certifications"
            ])
        
        # General tips for Indian market
        tips.extend([
            f"Tailor your resume for the Indian job market in {user_profile.get('location', 'your city')}",
            "Include any client-facing or leadership experience",
            "Mention language skills if relevant for the role",
            "Highlight any international exposure or remote work experience"
        ])
        
        return tips

    def generate_pdf(self, resume_content, template):
        """Generate PDF resume (placeholder - would use actual PDF library)"""
        # In production, use libraries like ReportLab or WeasyPrint
        # This is a placeholder that returns mock PDF data
        pdf_content = f"PDF Resume Generated - {template} template"
        return pdf_content.encode('utf-8')

    def _generate_personal_info(self, user_profile):
        """Generate personal information section"""
        return {
            'name': user_profile.get('name', ''),
            'email': user_profile.get('email', ''),
            'phone': user_profile.get('phone', ''),
            'location': user_profile.get('location', ''),
            'linkedin': user_profile.get('linkedin', ''),
            'github': user_profile.get('github', ''),
            'current_ctc': user_profile.get('current_ctc', ''),
            'expected_ctc': user_profile.get('expected_ctc', ''),
            'notice_period': user_profile.get('notice_period', '60 days')
        }

    def _generate_summary(self, user_profile, target_role):
        """Generate professional summary"""
        experience = user_profile.get('experience_years', 0)
        current_role = user_profile.get('current_role', 'Software Engineer')
        skills = user_profile.get('skills', [])
        
        summary = f"Experienced {current_role} with {experience} years of expertise in "
        
        if isinstance(skills, list):
            key_skills = skills[:3]  # Top 3 skills
            summary += ", ".join(key_skills)
        else:
            summary += "software development"
        
        summary += f". Seeking opportunities in {target_role or 'technology'} to leverage technical skills and drive innovation."
        
        return summary

    def _organize_skills(self, skills):
        """Organize skills into categories"""
        if not skills:
            return {}
        
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(',')]
        
        # Categorize skills
        categories = {
            'Programming Languages': [],
            'Frameworks & Libraries': [],
            'Databases': [],
            'Tools & Technologies': [],
            'Soft Skills': []
        }
        
        programming_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'scala']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']
        databases = ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle']
        tools = ['git', 'docker', 'kubernetes', 'jenkins', 'aws', 'gcp', 'azure']
        
        for skill in skills:
            skill_lower = skill.lower()
            if any(lang in skill_lower for lang in programming_langs):
                categories['Programming Languages'].append(skill)
            elif any(fw in skill_lower for fw in frameworks):
                categories['Frameworks & Libraries'].append(skill)
            elif any(db in skill_lower for db in databases):
                categories['Databases'].append(skill)
            elif any(tool in skill_lower for tool in tools):
                categories['Tools & Technologies'].append(skill)
            else:
                categories['Soft Skills'].append(skill)
        
        return {k: v for k, v in categories.items() if v}  # Remove empty categories

    def _format_experience(self, experience_list):
        """Format work experience"""
        formatted_exp = []
        
        for exp in experience_list:
            formatted_exp.append({
                'company': exp.get('company', ''),
                'role': exp.get('role', ''),
                'duration': f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}",
                'location': exp.get('location', ''),
                'achievements': exp.get('achievements', []),
                'technologies': exp.get('technologies', [])
            })
        
        return formatted_exp

    def _format_education(self, education):
        """Format education information"""
        return {
            'degree': education.get('degree', ''),
            'institution': education.get('institution', ''),
            'year': education.get('year', ''),
            'cgpa': education.get('cgpa', ''),
            'relevant_coursework': education.get('coursework', [])
        }

    def _format_projects(self, projects_list):
        """Format projects section"""
        formatted_projects = []
        
        for project in projects_list:
            formatted_projects.append({
                'name': project.get('name', ''),
                'description': project.get('description', ''),
                'technologies': project.get('technologies', []),
                'duration': project.get('duration', ''),
                'github_url': project.get('github_url', ''),
                'demo_url': project.get('demo_url', ''),
                'key_features': project.get('features', [])
            })
        
        return formatted_projects