from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)

# Load configuration
from app.config import config
app.config.from_object(config[os.getenv('FLASK_ENV', 'default')])

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Import services after app initialization
from app.services.job_search import JobSearchService
from app.services.nlp import analyze_user_query, generate_response_template
from app.services.user_service import UserService

# Initialize services
job_service = JobSearchService()
user_service = UserService()

# Routes
@app.route('/')
def index():
    """Render the main chatbot interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot interactions."""
    data = request.json
    user_message = data.get('message', '')
    user_id = session.get('user_id')
    
    # Analyze user query
    intent, params = analyze_user_query(user_message)
    
    response = {}
    
    if intent == 'job_search':
        # Get search parameters
        job_title = params.get('job_title', 'software engineer')
        location = params.get('location', 'remote')
        
        # Search for jobs
        jobs = job_service.search_jobs(job_title, location, limit=5)
        
        # Format response
        job_list = [job.to_dict() for job in jobs]
        response = {
            'message': f"I found {len(job_list)} {job_title} jobs in {location}.",
            'jobs': job_list
        }
    
    elif intent == 'salary_info':
        # Get salary information for job title
        job_title = params.get('job_title', 'software engineer')
        
        # This would typically come from a salary database or API
        # For demo purposes, we're using mock data
        salary_ranges = {
            'software engineer': '$80,000 - $120,000',
            'project manager': '$75,000 - $110,000',
            'data scientist': '$90,000 - $130,000',
            'marketing manager': '$65,000 - $95,000'
        }
        
        salary_range = salary_ranges.get(job_title.lower(), '$60,000 - $100,000')
        
        response = {
            'message': f"The typical salary range for a {job_title} is {salary_range}. Remember that you should negotiate for a competitive salary based on your experience and skills."
        }
    
    elif intent == 'company_info':
        # Get company information
        company_name = params.get('company', '')
        
        if company_name:
            # This would typically come from a company database or API
            # For demo purposes, we're using mock data
            response = {
                'message': f"Let me tell you about {company_name}. I'd need to get more information from specific company databases to provide accurate details."
            }
        else:
            response = {
                'message': "Here are some companies known for supporting women in the workplace: Adobe, American Express, Apple, Deloitte, IBM, Microsoft, Salesforce, and Zoom."
            }
    
    elif intent == 'resume_help':
        response = {
            'message': "I can help you improve your resume. Here are some tips specifically for women in the job market:\n\n" +
                      "1. Highlight your achievements with quantifiable results\n" +
                      "2. Use strong action verbs\n" +
                      "3. Include relevant keywords from job descriptions\n" +
                      "4. Showcase leadership experience even from non-work contexts\n" +
                      "5. List technical skills prominently\n\n" +
                      "Would you like me to review your resume? You can upload it or paste the text."
        }
    
    elif intent == 'interview_prep':
        response = {
            'message': "Let's prepare for your interviews. Here are some commonly asked questions:\n\n" +
                      "1. Tell me about yourself\n" +
                      "2. Why are you interested in this role?\n" +
                      "3. How do you handle challenges or conflicts?\n" +
                      "4. What are your strengths and weaknesses?\n" +
                      "5. Where do you see yourself in five years?\n\n" +
                      "I can also provide strategies for addressing potentially biased questions. Which question would you like to practice?"
        }
    
    else:
        # Generate response based on intent
        response_message = generate_response_template(intent, params)
        response = {
            'message': response_message
        }
    
    return jsonify(response)

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    
    success, result = user_service.create_user(email, password, first_name, last_name)
    
    if success:
        session['user_id'] = result
        return jsonify({'success': True, 'user_id': result})
    else:
        return jsonify({'success': False, 'error': result}), 400

@app.route('/api/login', methods=['POST'])
def login():
    """Log in a user."""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    success, result = user_service.authenticate_user(email, password)
    
    if success:
        user = result
        session['user_id'] = user.id
        return jsonify({'success': True, 'user': user.to_dict()})
    else:
        return jsonify({'success': False, 'error': result}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    """Log out a user."""
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/profile', methods=['GET', 'PUT'])
def profile():
    """Get or update user profile."""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    if request.method == 'GET':
        user = user_service.get_user(user_id)
        if user:
            return jsonify({'success': True, 'user': user.to_dict()})
        else:
            return jsonify({'success': False, 'error': 'User not found'}), 404
    
    elif request.method == 'PUT':
        data = request.json
        success, result = user_service.update_user_profile(user_id, data)
        
        if success:
            return jsonify({'success': True, 'user': result.to_dict()})
        else:
            return jsonify({'success': False, 'error': result}), 400

@app.route('/api/jobs', methods=['GET'])
def jobs():
    """Search for jobs."""
    keywords = request.args.get('keywords', '')
    location = request.args.get('location', 'remote')
    limit = int(request.args.get('limit', 10))
    
    jobs = job_service.search_jobs(keywords, location, limit=limit)
    job_list = [job.to_dict() for job in jobs]
    
    return jsonify({'success': True, 'jobs': job_list})

@app.route('/api/recommendations', methods=['GET'])
def recommendations():
    """Get job recommendations for the current user."""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    user = user_service.get_user(user_id)
    
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    
    jobs = job_service.get_job_recommendations(user)
    job_list = [job.to_dict() for job in jobs]
    
    return jsonify({'success': True, 'jobs': job_list})

@app.route('/api/parse_resume', methods=['POST'])
def parse_resume():
    """Parse resume text."""
    data = request.json
    resume_text = data.get('text', '')
    
    if not resume_text:
        return jsonify({'success': False, 'error': 'No resume text provided'}), 400
    
    parsed_data = user_service.parse_resume(resume_text)
    
    return jsonify({'success': True, 'data': parsed_data})

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Run app
    app.run(debug=app.config['DEBUG'])