from datetime import datetime
from app import db

class Job(db.Model):
    """Job model for storing job listings."""
    __tablename__ = "jobs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100))
    description = db.Column(db.Text)
    requirements = db.Column(db.Text)
    
    # Job details
    job_type = db.Column(db.String(50))  # Full-time, Part-time, Contract, etc.
    salary_range = db.Column(db.String(100))
    is_remote = db.Column(db.Boolean, default=False)
    has_flexible_hours = db.Column(db.Boolean, default=False)
    
    # Female-friendly indicators
    diversity_statement = db.Column(db.Boolean, default=False)
    parental_benefits = db.Column(db.Boolean, default=False)
    gender_neutral_language = db.Column(db.Boolean, default=False)
    
    # System fields
    source = db.Column(db.String(50))  # Indeed, LinkedIn, etc.
    external_id = db.Column(db.String(100))
    url = db.Column(db.String(500))
    posted_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    bias_score = db.Column(db.Float)  # Lower is better (less biased)
    
    def __repr__(self):
        return f"<Job {self.title} at {self.company}>"
    
    def to_dict(self):
        """Convert job object to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'description': self.description,
            'requirements': self.requirements,
            'job_type': self.job_type,
            'salary_range': self.salary_range,
            'is_remote': self.is_remote,
            'has_flexible_hours': self.has_flexible_hours,
            'diversity_statement': self.diversity_statement,
            'parental_benefits': self.parental_benefits,
            'gender_neutral_language': self.gender_neutral_language,
            'source': self.source,
            'url': self.url,
            'posted_date': self.posted_date.isoformat() if self.posted_date else None,
            'bias_score': self.bias_score
        }