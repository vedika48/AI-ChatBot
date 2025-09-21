# app/app.py
from app import create_app
from datetime import datetime

app = create_app()

# Health check endpoint
@app.route('/health')
def health_check():
    try:
        if app.db is not None:
            # Test MongoDB connection
            app.db.command('ping')
            return {
                'status': 'healthy', 
                'service': 'Career Compass India API',
                'database': 'connected',
                'database_type': 'MongoDB'
            }
        else:
            return {
                'status': 'unhealthy', 
                'service': 'Career Compass India API',
                'database': 'disconnected',
                'error': 'MongoDB not configured'
            }, 500
    except Exception as e:
        return {
            'status': 'unhealthy', 
            'service': 'Career Compass India API',
            'database': 'disconnected',
            'error': str(e)
        }, 500

# Root endpoint
@app.route('/')
def root():
    return {
        'message': 'Welcome to Career Compass India API',
        'version': '1.0.0',
        'database': 'MongoDB',
        'status': 'operational',
        'endpoints': {
            'authentication': '/api/auth',
            'job_recommendations': '/api/jobs',
            'resume_services': '/api/resume',
            'chat_assistant': '/api/chat',
            'salary_insights': '/api/salary',
            'health_check': '/health'
        }
    }

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return {'error': 'Bad request', 'message': str(error)}, 400

@app.errorhandler(404)
def not_found(error):
    return {'error': 'Endpoint not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

if __name__ == '__main__':
    # Create indexes for better performance
    with app.app_context():
        if app.db is not None:
            try:
                # Create indexes for better performance
                app.db.users.create_index('email', unique=True)
                app.db.jobs.create_index('title')
                app.db.jobs.create_index('location')
                app.db.jobs.create_index('skills')
                app.db.resume_analyses.create_index('user_id')
                app.db.chat_messages.create_index('user_id')
                app.db.salary_data.create_index('position')
                app.db.salary_data.create_index('location')
                
                print("✅ MongoDB indexes created successfully")
                
            except Exception as e:
                print(f"⚠️  Database initialization warning: {e}")
    
    print("🚀 Career Compass India API starting on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)