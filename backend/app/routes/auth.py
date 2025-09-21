from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from bson import ObjectId

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'name', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        db = current_app.db
        users_collection = db.users
        
        # Check if user already exists
        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists'}), 400
        
        # Create new user
        user_data = {
            'email': data['email'],
            'name': data['name'],
            'password_hash': generate_password_hash(data['password']),
            'location': data.get('location', ''),
            'experience_years': data.get('experience_years', 0),
            'skills': data.get('skills', []),
            'current_role': data.get('current_role', ''),
            'target_role': data.get('target_role', ''),
            'salary_expectation': data.get('salary_expectation', 0),
            'created_at': current_app.get_current_timestamp(),
            'updated_at': current_app.get_current_timestamp()
        }
        
        # Insert user into MongoDB
        result = users_collection.insert_one(user_data)
        user_id = str(result.inserted_id)
        
        # Generate access token
        access_token = create_access_token(identity=user_id)
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': {
                'id': user_id,
                'email': user_data['email'],
                'name': user_data['name'],
                'location': user_data['location']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email', '')
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        db = current_app.db
        users_collection = db.users
        
        # Find user
        user = users_collection.find_one({'email': email})
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate access token
        access_token = create_access_token(identity=str(user['_id']))
        
        return jsonify({
            'access_token': access_token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'location': user['location'],
                'current_role': user.get('current_role', '')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/profile', methods=['GET', 'PUT'])
@jwt_required()
def profile():
    """Get or update user profile"""
    try:
        user_id = get_jwt_identity()
        db = current_app.db
        users_collection = db.users
        
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if request.method == 'GET':
            return jsonify({
                'user': {
                    'id': str(user['_id']),
                    'email': user['email'],
                    'name': user['name'],
                    'location': user.get('location', ''),
                    'experience_years': user.get('experience_years', 0),
                    'skills': user.get('skills', []),
                    'current_role': user.get('current_role', ''),
                    'target_role': user.get('target_role', ''),
                    'salary_expectation': user.get('salary_expectation', 0)
                }
            })
        
        elif request.method == 'PUT':
            data = request.get_json()
            
            # Update user fields
            update_data = {}
            if 'name' in data:
                update_data['name'] = data['name']
            if 'location' in data:
                update_data['location'] = data['location']
            if 'experience_years' in data:
                update_data['experience_years'] = data['experience_years']
            if 'skills' in data:
                update_data['skills'] = data['skills']
            if 'current_role' in data:
                update_data['current_role'] = data['current_role']
            if 'target_role' in data:
                update_data['target_role'] = data['target_role']
            if 'salary_expectation' in data:
                update_data['salary_expectation'] = data['salary_expectation']
            
            update_data['updated_at'] = current_app.get_current_timestamp()
            
            users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': update_data}
            )
            
            return jsonify({'message': 'Profile updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500