from datetime import datetime
from app import db

class User(db.Model):
    """User model for storing user related details."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    
    # Career-related fields
    current_job_title = db.Column(db.String(100))
    industry = db.Column(db.String(100))
    years_experience = db.Column(db.Integer)
    skills = db.Column(db.Text)  # Comma-separated skills
    
    # Job search preferences
    desired_job_titles = db.Column(db.Text)  # Comma-separated job titles
    desired_locations = db.Column(db.Text)  # Comma-separated locations
    min_salary = db.Column(db.Integer)
    remote_preference = db.Column(db.Boolean, default=False)
    flexible_hours_preference = db.Column(db.Boolean, default=False)
    
    # System fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.email}>"
    
    def to_dict(self):
        """Convert user object to dictionary."""
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'current_job_title': self.current_job_title,
            'industry': self.industry,
            'years_experience': self.years_experience,
            'skills': self.skills.split(',') if self.skills else [],
            'desired_job_titles': self.desired_job_titles.split(',') if self.desired_job_titles else [],
            'desired_locations': self.desired_locations.split(',') if self.desired_locations else [],
            'min_salary': self.min_salary,
            'remote_preference': self.remote_preference,
            'flexible_hours_preference': self.flexible_hours_preference
        }