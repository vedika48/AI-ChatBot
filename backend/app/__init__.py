# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from bson import ObjectId
import os
import json
from datetime import timedelta

# Custom JSON encoder to handle ObjectId serialization
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'career-compass-india-secret-key-2025')
    app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/career_compass_india')
    
    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-string')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
    
    # Initialize MongoDB
    try:
        mongo_client = MongoClient(app.config['MONGO_URI'])
        app.db = mongo_client.get_database()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        app.db = None
    
    # Initialize JWT
    JWTManager(app)
    
    # Set custom JSON encoder
    app.json_encoder = JSONEncoder
    
    # Enable CORS
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"])
    
    # Register blueprints
    from app.routes.auth import bp as auth_bp
    from app.routes.jobs import bp as jobs_bp
    from app.routes.resume import bp as resume_bp
    from app.routes.chat import bp as chat_bp
    from app.routes.salary import bp as salary_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(resume_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(salary_bp)
    
    return app