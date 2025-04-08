import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
from app import app

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Load gender-biased words list
with open(os.path.join(os.path.dirname(__file__), '../../data/job_keywords.json'), 'r') as f:
    job_data = json.load(f)
    GENDER_BIASED_WORDS = job_data.get('gender_biased_words', [])
    FEMALE_SUPPORTIVE_TERMS = job_data.get('female_supportive_terms', [])
    SKILL_KEYWORDS = job_data.get('skill_keywords', [])

# Load sentiment model
tokenizer = AutoTokenizer.from_pretrained(app.config['MODEL_NAME'])
model = AutoModelForSequenceClassification.from_pretrained(app.config['MODEL_NAME'])

def preprocess_text(text):
    """Preprocess text for NLP operations."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def detect_gender_bias(text):
    """
    Detect gender bias in text.
    Returns a score between 0 and 1, where higher values indicate more bias.
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Count gender-biased words
    biased_word_count = sum(1 for word in GENDER_BIASED_WORDS if word in processed_text)
    
    # Count female-supportive terms
    supportive_term_count = sum(1 for term in FEMALE_SUPPORTIVE_TERMS if term in processed_text)
    
    # Calculate bias score (between 0 and 1)
    # Higher score means more bias
    total_words = len(processed_text.split())
    if total_words == 0:
        return 0.5  # Default score for empty text
    
    bias_score = biased_word_count / (total_words + 1) - (supportive_term_count / (total_words + 1))
    
    # Normalize score between 0 and 1
    bias_score = max(0, min(1, (bias_score + 0.5)))
    
    return bias_score

def extract_job_keywords(text):
    """Extract relevant keywords from job description."""
    doc = nlp(text)
    
    # Extract skills and qualifications
    keywords = []
    
    # Extract skill keywords
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text.lower():
            keywords.append(skill)
    
    # Extract entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE', 'GPE']:
            keywords.append(ent.text)
    
    # Extract noun phrases (potential skills)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    relevant_phrases = [phrase for phrase in noun_phrases 
                        if any(skill in phrase.lower() for skill in SKILL_KEYWORDS)]
    keywords.extend(relevant_phrases)
    
    # Remove duplicates and limit length
    keywords = list(set(keywords))[:20]
    
    return keywords

def analyze_sentiment(text):
    """Analyze sentiment of text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=app.config['MAX_SEQUENCE_LENGTH'], truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probabilities[0][1].item()
    
    return positive_score

def analyze_user_query(query):
    """Analyze user query to determine intent."""
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Define intent categories and their keywords
    intents = {
        'job_search': ['find', 'search', 'looking', 'job', 'position', 'work', 'career', 'opportunity'],
        'salary_info': ['salary', 'pay', 'compensation', 'money', 'wage', 'earn'],
        'company_info': ['company', 'organization', 'employer', 'workplace', 'culture'],
        'skill_development': ['learn', 'skill', 'improve', 'develop', 'education', 'training', 'course'],
        'resume_help': ['resume', 'cv', 'cover letter', 'application'],
        'interview_prep': ['interview', 'prepare', 'question', 'answer'],
        'discrimination': ['discrimination', 'bias', 'sexism', 'inequality', 'harassment', 'gender']
    }
    
    # Calculate scores for each intent
    intent_scores = {}
    for intent, keywords in intents.items():
        score = sum(1 for keyword in keywords if keyword in processed_query)
        intent_scores[intent] = score
    
    # Find the intent with the highest score
    top_intent = max(intent_scores.items(), key=lambda x: x[1])
    
    # If no clear intent is found, default to job_search
    if top_intent[1] == 0:
        return 'job_search', {}
    
    # Extract entities for parameter filling
    doc = nlp(query)
    params = {}
    
    # Extract job titles
    job_titles = []
    for chunk in doc.noun_chunks:
        if any(word in ['job', 'position', 'role'] for word in [token.text.lower() for token in chunk]):
            # Get the job title (usually the adjectives and nouns before "job/position/role")
            title_tokens = [token for token in chunk if token.pos_ in ['ADJ', 'NOUN'] and token.text.lower() not in ['job', 'position', 'role']]
            if title_tokens:
                job_titles.append(' '.join([token.text for token in title_tokens]))
    
    if job_titles:
        params['job_title'] = job_titles[0]
    
    # Extract locations
    locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
    if locations:
        params['location'] = locations[0]
    
    return top_intent[0], params

def generate_response_template(intent, params=None):
    """Generate template response based on intent."""
    if params is None:
        params = {}
    
    templates = {
        'job_search': [
            "I'll help you find {job_title} jobs in {location}.",
            "Let me search for {job_title} positions in {location} for you.",
            "I'm looking for {job_title} opportunities in {location}."
        ],
        'salary_info': [
            "Let me find salary information for {job_title} roles.",
            "I can tell you about compensation for {job_title} positions.",
            "Here's what I know about pay for {job_title} jobs."
        ],
        'company_info': [
            "I'll find information about companies that are known for supporting women in the workplace.",
            "Let me tell you about some female-friendly employers.",
            "I can help you research companies with good diversity practices."
        ],
        'skill_development': [
            "Here are some resources to develop your skills in {skill_area}.",
            "I can suggest ways to improve your {skill_area} abilities.",
            "Let me help you find learning opportunities for {skill_area}."
        ],
        'resume_help': [
            "I can help you improve your resume to highlight your strengths.",
            "Let me give you some tips to make your resume stand out.",
            "I'll help you craft a resume that gets past automated screening systems."
        ],
        'interview_prep': [
            "Let's prepare for your upcoming interviews with some practice questions.",
            "I can help you get ready for common interview questions in your field.",
            "Let me suggest some strategies for your interview preparation."
        ],
        'discrimination': [
            "I understand you're concerned about workplace discrimination. Here's some information that might help.",
            "Let me provide some resources about addressing gender bias in the workplace.",
            "I can help you understand your rights regarding gender discrimination."
        ]
    }
    
    # Get templates for the intent
    intent_templates = templates.get(intent, templates['job_search'])
    
    # Select a template
    import random
    template = random.choice(intent_templates)
    
    # Fill in parameters
    for key, value in params.items():
        placeholder = '{' + key + '}'
        if placeholder in template:
            template = template.replace(placeholder, value)
        else:
            # Handle missing parameters in template
            template = template.replace('{' + key + '}', '').strip()
    
    # Clean up any remaining unfilled placeholders
    template = re.sub(r'\{[^}]*\}', '', template).strip()
    
    return template