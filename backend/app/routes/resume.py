from flask import Blueprint, request, jsonify, current_app
from app.ml_models.resume_optimizer import ResumeOptimizer
from app.services.resume_builder import ResumeBuilder
from bson import ObjectId
import base64
from io import BytesIO

bp = Blueprint('resume', __name__, url_prefix='/api/resume')

resume_optimizer = ResumeOptimizer()
resume_builder = ResumeBuilder()

@bp.route('/analyze', methods=['POST'])
def analyze_resume():
    """Analyze resume and provide optimization suggestions"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        job_description = data.get('job_description', '')
        user_id = data.get('user_id')
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
            
        # Analyze resume using ML model
        analysis = resume_optimizer.analyze_resume(resume_text, job_description)
        
        # Get additional insights
        ats_score = resume_builder.calculate_ats_score(resume_text)
        keyword_density = resume_builder.analyze_keyword_density(resume_text, job_description)
        
        # Save analysis to MongoDB
        if user_id:
            db = current_app.db
            resume_analysis_collection = db.resume_analyses
            
            analysis_doc = {
                'user_id': ObjectId(user_id),
                'resume_text': resume_text,
                'job_description': job_description,
                'analysis_result': analysis,
                'ats_score': ats_score,
                'keyword_density': keyword_density,
                'analyzed_at': current_app.get_current_timestamp()
            }
            resume_analysis_collection.insert_one(analysis_doc)
        
        return jsonify({
            'match_score': analysis['match_score'],
            'ats_score': ats_score,
            'suggestions': analysis['suggestions'],
            'missing_keywords': analysis['missing_keywords'],
            'keyword_density': keyword_density,
            'sections_analysis': resume_builder.analyze_sections(resume_text),
            'formatting_tips': resume_builder.get_formatting_tips()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/build', methods=['POST'])
def build_resume():
    """Build a resume based on user profile"""
    try:
        data = request.get_json()
        user_profile = data.get('user_profile', {})
        template = data.get('template', 'professional')
        target_role = data.get('target_role', '')
        user_id = data.get('user_id')
        
        # Generate resume content
        resume_content = resume_builder.generate_resume(user_profile, template, target_role)
        
        # Generate PDF
        pdf_data = resume_builder.generate_pdf(resume_content, template)
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Save generated resume to MongoDB
        if user_id:
            db = current_app.db
            generated_resumes_collection = db.generated_resumes
            
            resume_doc = {
                'user_id': ObjectId(user_id),
                'template': template,
                'target_role': target_role,
                'resume_content': resume_content,
                'generated_at': current_app.get_current_timestamp(),
                'download_count': 0
            }
            generated_resumes_collection.insert_one(resume_doc)
        
        return jsonify({
            'resume_content': resume_content,
            'pdf_data': pdf_base64,
            'template_used': template,
            'optimization_tips': resume_builder.get_optimization_tips(user_profile, target_role)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/templates', methods=['GET'])
def get_templates():
    """Get available resume templates"""
    templates = [
        {
            'id': 'professional',
            'name': 'Professional',
            'description': 'Clean and formal design suitable for corporate roles',
            'preview_url': '/static/templates/professional_preview.png',
            'suitable_for': ['Corporate', 'Banking', 'Consulting']
        },
        {
            'id': 'modern',
            'name': 'Modern',
            'description': 'Contemporary design with color accents for creative roles',
            'preview_url': '/static/templates/modern_preview.png',
            'suitable_for': ['Tech', 'Design', 'Startup']
        },
        {
            'id': 'minimal',
            'name': 'Minimal',
            'description': 'Simple and clean layout focusing on content',
            'preview_url': '/static/templates/minimal_preview.png',
            'suitable_for': ['Academia', 'Research', 'Government']
        }
    ]
    
    return jsonify({'templates': templates})

@bp.route('/history/<user_id>', methods=['GET'])
def get_resume_history(user_id):
    """Get user's resume generation history"""
    try:
        db = current_app.db
        generated_resumes_collection = db.generated_resumes
        
        resumes = list(generated_resumes_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('generated_at', -1).limit(10))
        
        for resume in resumes:
            resume['_id'] = str(resume['_id'])
            resume['user_id'] = str(resume['user_id'])
        
        return jsonify({'resume_history': resumes})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500