from flask import Blueprint, request, jsonify, current_app
from app.services.chat_service import ChatService
from app.ml_models.job_matcher import RealTimeJobMatcher
from app.ml_models.resume_optimizer import ResumeOptimizer
from app.ml_models.salary_predictor import SalaryPredictor, SalaryDataCollector, MarketAnalyzer
from bson import ObjectId
import json
from datetime import datetime, timezone  # Added timezone import

bp = Blueprint('chat', __name__, url_prefix='/api/chat')

chat_service = ChatService()
job_matcher = RealTimeJobMatcher()
resume_optimizer = ResumeOptimizer()
data_collector = SalaryDataCollector()
market_analyzer = MarketAnalyzer(data_collector)
salary_predictor = SalaryPredictor(data_collector, market_analyzer)

@bp.route('/message', methods=['POST'])
def send_message():
    """Handle chat messages from the frontend"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id')
        context = data.get('context', {})
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Save chat message to MongoDB
        db = current_app.db
        chat_collection = db.chat_messages
        
        # Use timezone-aware datetime
        current_timestamp = datetime.now(timezone.utc)
        
        chat_doc = {
            'user_id': ObjectId(user_id) if user_id else None,
            'message': user_message,
            'context': context,
            'timestamp': current_timestamp,
            'type': 'user'
        }
        
        chat_collection.insert_one(chat_doc)
            
        # Process the message and generate response
        response = chat_service.process_message(user_message, user_id, context)
        
        # Save bot response to MongoDB
        bot_response_doc = {
            'user_id': ObjectId(user_id) if user_id else None,
            'message': response.get('response', ''),
            'context': response.get('context', {}),
            'timestamp': current_timestamp,  # Use the same timestamp or create a new one
            'type': 'bot'
        }
        
        chat_collection.insert_one(bot_response_doc)
        
        return jsonify({
            'response': response,
            'suggestions': chat_service.get_follow_up_suggestions(user_message),
            'timestamp': current_timestamp.isoformat()  # Return as ISO format string
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Chat Error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested queries for the chat interface"""
    suggestions = [
        "Find software engineer jobs in Bangalore",
        "How to prepare for a technical interview at Indian companies",
        "Companies with good maternity leave policies in India",
        "Salary negotiation tips for women in tech",
        "Remote work opportunities for Indian professionals",
        "Best cities in India for software engineers",
        "How to switch from service company to product company",
        "Interview experiences at Indian unicorns",
        "Work-life balance in Indian IT companies",
        "Career growth opportunities for women in tech"
    ]
    
    return jsonify({'suggestions': suggestions})

@bp.route('/history/<user_id>', methods=['GET'])
def get_chat_history(user_id):
    """Get chat history for a user"""
    try:
        db = current_app.db
        chat_collection = db.chat_messages
        
        chat_history = list(chat_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('timestamp', 1).limit(50))
        
        # Convert ObjectId to string for JSON serialization
        for chat in chat_history:
            chat['_id'] = str(chat['_id'])
            if 'user_id' in chat and chat['user_id']:
                chat['user_id'] = str(chat['user_id'])
            # Convert datetime to ISO format string for JSON serialization
            if 'timestamp' in chat and isinstance(chat['timestamp'], datetime):
                chat['timestamp'] = chat['timestamp'].isoformat()
        
        return jsonify({'chat_history': chat_history})
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Chat History Error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500