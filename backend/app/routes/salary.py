from flask import Blueprint, request, jsonify, current_app
from app.ml_models.salary_predictor import SalaryPredictor, SalaryDataCollector, MarketAnalyzer
from app.services.salary_data import SalaryDataService
from bson import ObjectId

bp = Blueprint('salary', __name__, url_prefix='/api/salary')

data_collector = SalaryDataCollector()
market_analyzer = MarketAnalyzer(data_collector)
salary_predictor = SalaryPredictor(data_collector, market_analyzer)
salary_data_service = SalaryDataService()

@bp.route('/predict', methods=['POST'])
def predict_salary():
    """Predict salary based on user profile"""
    try:
        data = request.get_json()
        user_profile = data.get('user_profile', {})
        user_id = data.get('user_id')
        
        # Get ML prediction
        prediction = salary_predictor.predict_salary(user_profile)
        
        # Get market data
        market_data = salary_data_service.get_market_data(
            role=user_profile.get('role', ''),
            location=user_profile.get('location', 'bangalore'),
            experience=user_profile.get('experience_years', 0)
        )
        
        # Get negotiation insights
        negotiation_tips = salary_data_service.get_negotiation_tips(
            current_salary=user_profile.get('current_salary', 0),
            target_salary=prediction['predicted_salary'],
            user_profile=user_profile
        )
        
        # Save prediction to MongoDB
        if user_id:
            db = current_app.db
            salary_predictions_collection = db.salary_predictions
            
            prediction_doc = {
                'user_id': ObjectId(user_id),
                'user_profile': user_profile,
                'prediction': prediction,
                'market_data': market_data,
                'predicted_at': current_app.get_current_timestamp()
            }
            salary_predictions_collection.insert_one(prediction_doc)
        
        return jsonify({
            'predicted_salary': prediction['predicted_salary'],
            'salary_range': {
                'min': prediction['min_range'],
                'max': prediction['max_range']
            },
            'market_data': market_data,
            'negotiation_tips': negotiation_tips,
            'confidence': prediction.get('confidence', 0.8)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/benchmark', methods=['POST'])
def get_salary_benchmark():
    """Get salary benchmark data"""
    try:
        data = request.get_json()
        role = data.get('role', '')
        location = data.get('location', 'bangalore')
        experience = data.get('experience', 0)
        company_size = data.get('company_size', 'medium')
        
        benchmark_data = salary_data_service.get_benchmark_data(
            role=role,
            location=location,
            experience=experience,
            company_size=company_size
        )
        
        return jsonify({
            'benchmark_data': benchmark_data,
            'percentiles': {
                'p25': benchmark_data['percentile_25'],
                'p50': benchmark_data['percentile_50'],
                'p75': benchmark_data['percentile_75'],
                'p90': benchmark_data['percentile_90']
            },
            'location_comparison': salary_data_service.get_location_comparison(role, experience),
            'growth_trends': salary_data_service.get_growth_trends(role, location)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/submissions', methods=['POST'])
def submit_salary_data():
    """Allow users to submit salary data (anonymously)"""
    try:
        data = request.get_json()
        
        db = current_app.db
        salary_submissions_collection = db.salary_submissions
        
        submission_data = {
            'role': data.get('role'),
            'company': data.get('company'),
            'location': data.get('location'),
            'experience': data.get('experience'),
            'salary': data.get('salary'),
            'bonus': data.get('bonus', 0),
            'equity': data.get('equity', 0),
            'submitted_at': current_app.get_current_timestamp(),
            'is_anonymous': data.get('is_anonymous', True)
        }
        
        salary_submissions_collection.insert_one(submission_data)
        
        return jsonify({'message': 'Salary data submitted successfully (anonymously)'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500