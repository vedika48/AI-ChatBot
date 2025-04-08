from app.models.user import User
from app import db
import hashlib
import os

class UserService:
    """Service for managing user accounts and profiles."""
    
    def create_user(self, email, password, first_name, last_name):
        """Create a new user."""
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return False, "Email already registered"
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user
        user = User(
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            return True, user.id
        except Exception as e:
            db.session.rollback()
            return False, str(e)
    
    def authenticate_user(self, email, password):
        """Authenticate a user."""
        user = User.query.filter_by(email=email).first()
        
        if not user:
            return False, "User not found"
        
        # Check password
        password_hash = self._hash_password(password)
        if user.password_hash != password_hash:
            return False, "Invalid password"
        
        return True, user
    
    def update_user_profile(self, user_id, profile_data):
        """Update user profile."""
        user = User.query.get(user_id)
        
        if not user:
            return False, "User not found"
        
        try:
            # Update allowed fields
            allowed_fields = [
                'first_name', 'last_name', 'current_job_title', 
                'industry', 'years_experience', 'skills',
                'desired_job_titles', 'desired_locations', 'min_salary',
                'remote_preference', 'flexible_hours_preference'
            ]
            
            for field in allowed_fields:
                if field in profile_data:
                    setattr(user, field, profile_data[field])
            
            db.session.commit()
            return True, user
        except Exception as e:
            db.session.rollback()
            return False, str(e)
    
    def get_user(self, user_id):
        """Get user by ID."""
        user = User.query.get(user_id)
        return user
    
    def _hash_password(self, password):
        """Hash password using SHA-256."""
        # In production, use a proper password hashing library like bcrypt
        return hashlib.sha256(password.encode()).hexdigest()
    
    def parse_resume(self, resume_text):
        """Extract relevant information from resume text."""
        # This is a simplified implementation - in a real-world scenario,
        # you would use more sophisticated NLP techniques
        
        # Extract skills
        skill_keywords = ['Python', 'Java', 'JavaScript', 'SQL', 'Excel', 
                         'Leadership', 'Communication', 'Project Management']
        skills = []
        
        for skill in skill_keywords:
            if skill.lower() in resume_text.lower():
                skills.append(skill)
        
        # Extract job titles
        job_title_patterns = [
            r'([A-Za-z\s]+) Engineer',
            r'([A-Za-z\s]+) Manager',
            r'([A-Za-z\s]+) Developer',
            r'([A-Za-z\s]+) Specialist',
            r'([A-Za-z\s]+) Analyst'
        ]
        
        import re
        job_titles = []
        
        for pattern in job_title_patterns:
            matches = re.findall(pattern, resume_text)
            job_titles.extend(matches)
        
        # Extract years of experience
        experience_patterns = [
            r'(\d+)[\+]? years of experience',
            r'(\d+)[\+]? years experience',
            r'experience of (\d+)[\+]? years'
        ]
        
        years_experience = None
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text)
            if matches:
                years_experience = int(matches[0])
                break
        
        return {
            'skills': skills,
            'job_titles': job_titles,
            'years_experience': years_experience
        }