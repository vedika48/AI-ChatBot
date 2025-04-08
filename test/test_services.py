import unittest
import json
import os
from unittest.mock import patch, MagicMock

# Import the services to test
from app.services.job_search import JobSearchService
from app.services.nlp import NLPService
from app.services.user_service import UserService
from app.models.user import User
from app.models.job import Job

class TestJobSearchService(unittest.TestCase):
    """Test cases for the Job Search Service"""
    
    def setUp(self):
        # Set up test data
        self.job_search = JobSearchService()
        self.test_query = {
            "title": "Software Engineer",
            "location": "San Francisco",
            "remote": True,
            "keywords": ["python", "react", "women in tech"]
        }
        
        # Mock job data
        self.test_jobs = [
            {
                "id": "job1",
                "title": "Senior Software Engineer",
                "company": "TechWomen Inc.",
                "location": "San Francisco, CA",
                "remote": True,
                "description": "Join our diverse engineering team working on cutting-edge AI solutions.",
                "requirements": ["Python", "React", "5+ years experience"],
                "benefits": ["Flexible work", "Parental leave", "Mentorship program"],
                "is_female_friendly": True
            },
            {
                "id": "job2",
                "title": "Full Stack Developer",
                "company": "WebCorp",
                "location": "San Francisco, CA",
                "remote": False,
                "description": "Build scalable web applications for our clients.",
                "requirements": ["JavaScript", "React", "Node.js"],
                "benefits": ["Health insurance", "401k"],
                "is_female_friendly": False
            }
        ]
    
    @patch('app.services.job_search.requests.get')
    def test_search_jobs(self, mock_get):
        """Test that job search returns expected results"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": self.test_jobs}
        mock_get.return_value = mock_response
        
        # Call the method
        results = self.job_search.search_jobs(self.test_query)
        
        # Verify the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Senior Software Engineer")
        self.assertTrue(results[0]["is_female_friendly"])
        self.assertEqual(results[1]["title"], "Full Stack Developer")
    
    @patch('app.services.job_search.requests.get')
    def test_filter_female_friendly(self, mock_get):
        """Test filtering jobs for female-friendly workplaces"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": self.test_jobs}
        mock_get.return_value = mock_response
        
        # Call the method with female-friendly filter
        query_with_filter = self.test_query.copy()
        query_with_filter["female_friendly_only"] = True
        results = self.job_search.search_jobs(query_with_filter)
        
        # Verify only female-friendly jobs are returned
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Senior Software Engineer")
        self.assertTrue(results[0]["is_female_friendly"])
    
    def test_get_job_details(self):
        """Test fetching details for a specific job"""
        # Mock job details
        job_id = "job1"
        job_details = self.test_jobs[0]
        
        # Mock the get_job_details method
        with patch.object(self.job_search, '_fetch_job_details', return_value=job_details):
            result = self.job_search.get_job_details(job_id)
            
            # Verify the result
            self.assertEqual(result["id"], job_id)
            self.assertEqual(result["title"], "Senior Software Engineer")
            self.assertEqual(result["company"], "TechWomen Inc.")


class TestNLPService(unittest.TestCase):
    """Test cases for the NLP Service"""
    
    def setUp(self):
        # Set up test data
        self.nlp_service = NLPService()
        
        # Test resume text
        self.resume_text = """
        Jane Doe
        San Francisco, CA | jane.doe@email.com | (123) 456-7890
        
        EXPERIENCE
        Senior Software Engineer, TechCorp (2018-Present)
        - Led development of Python microservices architecture
        - Mentored junior engineers on React best practices
        - Improved test coverage from 65% to 92%
        
        Software Developer, WebSolutions (2015-2018)
        - Developed responsive web applications using JavaScript and React
        - Collaborated with UX designers to implement intuitive interfaces
        
        EDUCATION
        B.S. Computer Science, University of California, Berkeley (2015)
        
        SKILLS
        Python, JavaScript, React, SQL, AWS, Docker, Git
        """
        
        # Test job description
        self.job_description = """
        We're looking for a Senior Software Engineer with experience in Python and React.
        The ideal candidate will have 5+ years of experience and be comfortable with microservices.
        Our company values diversity and provides mentorship opportunities.
        """
    
    def test_extract_skills(self):
        """Test skill extraction from resume"""
        # Call the method
        skills = self.nlp_service.extract_skills(self.resume_text)
        
        # Verify extracted skills
        expected_skills = ["Python", "JavaScript", "React", "SQL", "AWS", "Docker", "Git"]
        for skill in expected_skills:
            self.assertIn(skill, skills)
    
    def test_extract_experience(self):
        """Test experience extraction from resume"""
        # Call the method
        experience = self.nlp_service.extract_experience(self.resume_text)
        
        # Verify extracted experience
        self.assertIn("Senior Software Engineer", experience["job_titles"])
        self.assertIn("Software Developer", experience["job_titles"])
        self.assertGreaterEqual(experience["years"], 5)  # 2015-Present should be 5+ years
    
    def test_match_score(self):
        """Test job matching score calculation"""
        # Call the method
        score = self.nlp_service.calculate_match_score(self.resume_text, self.job_description)
        
        # Verify the score is within expected range (0-100)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        # The score should be high since there's good overlap
        self.assertGreaterEqual(score, 70)
    
    def test_generate_improvement_suggestions(self):
        """Test generating resume improvement suggestions"""
        # Call the method
        suggestions = self.nlp_service.generate_improvement_suggestions(
            self.resume_text, self.job_description
        )
        
        # Verify suggestions are generated
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


class TestUserService(unittest.TestCase):
    """Test cases for the User Service"""
    
    def setUp(self):
        # Set up test data
        self.user_service = UserService()
        self.test_user_data = {
            "email": "test@example.com",
            "password": "securePassword123",
            "first_name": "Jane",
            "last_name": "Doe",
            "current_job_title": "Software Engineer",
            "desired_job_titles": ["Senior Software Engineer", "Lead Developer"],
            "skills": ["Python", "JavaScript", "React"],
            "desired_locations": ["San Francisco", "Remote"],
            "min_salary": 120000,
            "remote_preference": True,
            "flexible_hours_preference": True
        }
    
    @patch('app.services.user_service.db.session')
    def test_create_user(self, mock_db_session):
        """Test user creation"""
        # Call the method
        user = self.user_service.create_user(
            self.test_user_data["email"],
            self.test_user_data["password"],
            self.test_user_data["first_name"],
            self.test_user_data["last_name"]
        )
        
        # Verify user is created with correct attributes
        self.assertIsInstance(user, User)
        self.assertEqual(user.email, self.test_user_data["email"])
        self.assertEqual(user.first_name, self.test_user_data["first_name"])
        self.assertEqual(user.last_name, self.test_user_data["last_name"])
        
        # Verify password is hashed
        self.assertNotEqual(user.password_hash, self.test_user_data["password"])
        
        # Verify db session called
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.services.user_service.db.session')
    def test_update_profile(self, mock_db_session):
        """Test profile update"""
        # Create a mock user
        user = User(
            id=1,
            email=self.test_user_data["email"],
            first_name=self.test_user_data["first_name"],
            last_name=self.test_user_data["last_name"]
        )
        
        # Mock the get_user_by_id method
        with patch.object(self.user_service, 'get_user_by_id', return_value=user):
            # Call the method
            updated_user = self.user_service.update_profile(
                user_id=1,
                profile_data={
                    "current_job_title": self.test_user_data["current_job_title"],
                    "desired_job_titles": self.test_user_data["desired_job_titles"],
                    "skills": self.test_user_data["skills"],
                    "desired_locations": self.test_user_data["desired_locations"],
                    "min_salary": self.test_user_data["min_salary"],
                    "remote_preference": self.test_user_data["remote_preference"],
                    "flexible_hours_preference": self.test_user_data["flexible_hours_preference"]
                }
            )
            
            # Verify user is updated with correct attributes
            self.assertEqual(updated_user.current_job_title, self.test_user_data["current_job_title"])
            self.assertEqual(updated_user.desired_job_titles, self.test_user_data["desired_job_titles"])
            self.assertEqual(updated_user.skills, self.test_user_data["skills"])
            self.assertEqual(updated_user.min_salary, self.test_user_data["min_salary"])
            self.assertTrue(updated_user.remote_preference)
            
            # Verify db session called
            mock_db_session.commit.assert_called_once()
    
    @patch('app.services.user_service.db.session')
    def test_save_job(self, mock_db_session):
        """Test saving a job for a user"""
        # Create a mock user and job
        user = User(id=1, email=self.test_user_data["email"])
        job = Job(id="job1", title="Senior Software Engineer")
        
        # Mock the get_user_by_id method
        with patch.object(self.user_service, 'get_user_by_id', return_value=user):
            # Mock the get_job_by_id method
            with patch('app.services.user_service.Job.query') as mock_job_query:
                mock_job_query.filter_by.return_value.first.return_value = job
                
                # Call the method
                result = self.user_service.save_job_for_user(user_id=1, job_id="job1")
                
                # Verify job is saved to user's saved_jobs
                self.assertTrue(result)
                self.assertIn(job, user.saved_jobs)
                
                # Verify db session called
                mock_db_session.commit.assert_called_once()
    
    def test_authenticate_user_valid(self):
        """Test user authentication with valid credentials"""
        # Create a mock user with hashed password
        user = User(
            email=self.test_user_data["email"],
            password_hash=self.user_service._hash_password(self.test_user_data["password"])
        )
        
        # Mock the get_user_by_email method
        with patch.object(self.user_service, 'get_user_by_email', return_value=user):
            # Call the method
            authenticated_user = self.user_service.authenticate_user(
                self.test_user_data["email"],
                self.test_user_data["password"]
            )
            
            # Verify authentication successful
            self.assertIsNotNone(authenticated_user)
            self.assertEqual(authenticated_user.email, self.test_user_data["email"])
    
    def test_authenticate_user_invalid(self):
        """Test user authentication with invalid credentials"""
        # Create a mock user with hashed password
        user = User(
            email=self.test_user_data["email"],
            password_hash=self.user_service._hash_password(self.test_user_data["password"])
        )
        
        # Mock the get_user_by_email method
        with patch.object(self.user_service, 'get_user_by_email', return_value=user):
            # Call the method with wrong password
            authenticated_user = self.user_service.authenticate_user(
                self.test_user_data["email"],
                "wrongPassword"
            )
            
            # Verify authentication failed
            self.assertIsNone(authenticated_user)


if __name__ == '__main__':
    unittest.main()