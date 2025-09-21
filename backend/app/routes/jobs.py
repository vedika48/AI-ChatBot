from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from app.ml_models.job_matcher import SyncJobFetcher
from bson import ObjectId

bp = Blueprint('jobs', __name__, url_prefix='/api/jobs')

# Initialize the sync fetcher
sync_fetcher = SyncJobFetcher()

@bp.route('/recommendations', methods=['POST'])
def get_job_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        db = current_app.db
        users_collection = db.users
        
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        user_profile = {
            'skills': user.get('skills', []),
            'experience_years': user.get('experience_years', 0),
            'location': user.get('location', ''),
            'current_role': user.get('current_role', '')
        }
        
        # Use synchronous method
        recommendations = sync_fetcher.recommend_jobs_sync(user_profile)
        
        # Save recommendations to MongoDB
        job_recommendations_collection = db.job_recommendations
        
        for rec in recommendations:
            job_rec = {
                'user_id': ObjectId(user_id),
                'job_title': rec['job_title'],
                'company': rec['company'],
                'location': rec['location'],
                'salary_range': rec['salary_range'],
                'match_score': rec['match_score'],
                'job_url': rec['job_url'],
                'recommended_at': datetime.now(),
                'viewed': False,
                'applied': False
            }
            job_recommendations_collection.insert_one(job_rec)
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/search', methods=['POST'])
def search_jobs():
    try:
        data = request.get_json()
        query = data.get('query', '')
        location = data.get('location', '')
        
        # Use synchronous method
        jobs = sync_fetcher.fetch_jobs_sync(query, location, limit=20)
        
        return jsonify({'jobs': jobs, 'count': len(jobs)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@bp.route('/saved', methods=['GET'])
def get_saved_jobs():
    """Get user's saved jobs"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
            
        db = current_app.db
        saved_jobs_collection = db.saved_jobs
        
        saved_jobs = list(saved_jobs_collection.find(
            {'user_id': ObjectId(user_id)}
        ))
        
        for job in saved_jobs:
            job['_id'] = str(job['_id'])
            job['user_id'] = str(job['user_id'])
        
        return jsonify({'saved_jobs': saved_jobs})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500