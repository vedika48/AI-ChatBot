import re
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import os
from collections import Counter
from app import app

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load spaCy models for enhanced NLP capabilities
try:
    nlp = spacy.load('en_core_web_lg')  # Larger model with better vectors
except OSError:
    # Fallback to medium model if large isn't available
    nlp = spacy.load('en_core_web_md')

# Load job-related data with more comprehensive categorization
data_path = os.path.join(os.path.dirname(__file__), '../../data/job_keywords.json')
with open(data_path, 'r') as f:
    job_data = json.load(f)
    
    # Extract all keyword categories
    GENDER_BIASED_WORDS = job_data.get('gender_biased_words', [])
    MASCULINE_CODED_WORDS = job_data.get('masculine_coded_words', [])
    FEMININE_CODED_WORDS = job_data.get('feminine_coded_words', [])
    FEMALE_SUPPORTIVE_TERMS = job_data.get('female_supportive_terms', [])
    INCLUSIVE_LANGUAGE = job_data.get('inclusive_language', [])
    SKILL_KEYWORDS = job_data.get('skill_keywords', [])
    INDUSTRY_TERMS = job_data.get('industry_terms', {})
    ROLE_HIERARCHIES = job_data.get('role_hierarchies', {})

# Load pre-trained models
# Sentiment analysis model
sentiment_tokenizer = AutoTokenizer.from_pretrained(app.config['MODEL_NAME'])
sentiment_model = AutoModelForSequenceClassification.from_pretrained(app.config['MODEL_NAME'])

# Sentence embeddings model for semantic similarity
embedding_model_name = app.config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Cache for embeddings to improve performance
embedding_cache = {}

def get_sentence_embedding(text):
    """Generate embeddings for text using pre-trained transformer model"""
    if text in embedding_cache:
        return embedding_cache[text]
    
    # Tokenize and prepare input
    inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, 
                                max_length=app.config.get('MAX_SEQUENCE_LENGTH', 512))
    
    # Get embeddings
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    
    # Use mean pooling to get sentence embedding
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Convert to numpy and cache
    result = embedding.squeeze().numpy()
    embedding_cache[text] = result
    return result

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Enhanced preprocessing for text with options for stopword removal and lemmatization.
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers but preserve sentence structure
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        # Remove stopwords but preserve some important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'but', 'however', 'although', 'though'}
        tokens = [word for word in tokens if word not in stop_words]
    
    if lemmatize:
        # Lemmatize tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def detect_gender_bias(text, detailed=False):
    """
    Advanced gender bias detection with contextual analysis and detailed reporting.
    Returns a bias score and optionally detailed analysis.
    """
    # Get both raw and preprocessed text for different analysis aspects
    raw_text = text.lower()
    processed_text = preprocess_text(text)
    
    # Process text with spaCy for contextual analysis
    doc = nlp(text)
    
    # Basic count analysis
    masculine_word_count = sum(1 for word in MASCULINE_CODED_WORDS if word in raw_text)
    feminine_word_count = sum(1 for word in FEMININE_CODED_WORDS if word in raw_text)
    biased_word_count = sum(1 for word in GENDER_BIASED_WORDS if word in raw_text)
    supportive_term_count = sum(1 for term in FEMALE_SUPPORTIVE_TERMS if term in raw_text)
    inclusive_term_count = sum(1 for term in INCLUSIVE_LANGUAGE if term in raw_text)
    
    # Contextual analysis - check for negation around terms
    # Analyze noun-adjective relationships and other contextual aspects
    contextual_factors = 0
    gender_phrases = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Check for negative contexts around positive terms
        if any(term in sent_text for term in FEMALE_SUPPORTIVE_TERMS):
            if any(neg in sent_text for neg in ["not", "no", "never", "without", "lacks", "lack of"]):
                contextual_factors -= 1
                gender_phrases.append(sent.text)
        
        # Check for conditional language around inclusive terms
        if any(term in sent_text for term in INCLUSIVE_LANGUAGE):
            if any(cond in sent_text for cond in ["if", "when", "might", "could", "should"]):
                contextual_factors -= 0.5
        
        # Identify phrases with gender bias
        for chunk in sent.noun_chunks:
            if any(word.text.lower() in GENDER_BIASED_WORDS for word in chunk):
                gender_phrases.append(chunk.text)
    
    # Word frequency analysis - compare distribution of masculine vs feminine terms
    total_words = len(processed_text.split())
    if total_words == 0:
        return 0.5, {} if detailed else 0.5
    
    # Calculate base bias score
    masculine_ratio = masculine_word_count / (total_words + 1)
    feminine_ratio = feminine_word_count / (total_words + 1)
    biased_ratio = biased_word_count / (total_words + 1)
    supportive_ratio = supportive_term_count / (total_words + 1)
    inclusive_ratio = inclusive_term_count / (total_words + 1)
    
    # Calculate gender skew (positive values indicate masculine skew)
    gender_skew = masculine_ratio - feminine_ratio
    
    # Calculate final bias score with contextual adjustments
    bias_score = biased_ratio - supportive_ratio - inclusive_ratio + (gender_skew * 0.5) + (contextual_factors * 0.1)
    
    # Normalize score between 0 and 1 (0 = unbiased, 1 = highly biased)
    normalized_bias_score = max(0, min(1, (bias_score + 0.5)))
    
    if not detailed:
        return normalized_bias_score
    
    # Prepare detailed report
    report = {
        "bias_score": normalized_bias_score,
        "masculine_terms": {word: raw_text.count(word) for word in MASCULINE_CODED_WORDS if word in raw_text},
        "feminine_terms": {word: raw_text.count(word) for word in FEMININE_CODED_WORDS if word in raw_text},
        "biased_terms": {word: raw_text.count(word) for word in GENDER_BIASED_WORDS if word in raw_text},
        "supportive_terms": {term: raw_text.count(term) for term in FEMALE_SUPPORTIVE_TERMS if term in raw_text},
        "inclusive_terms": {term: raw_text.count(term) for term in INCLUSIVE_LANGUAGE if term in raw_text},
        "gender_skew": gender_skew,
        "contextual_factors": contextual_factors,
        "gender_phrases": gender_phrases,
        "recommendations": []
    }
    
    # Generate recommendations based on findings
    if normalized_bias_score > 0.7:
        report["recommendations"].append("Consider reducing use of masculine-coded language")
        
    if gender_skew > 0.2:
        report["recommendations"].append("Text appears to favor masculine-coded terms")
        
    if biased_word_count > 0:
        report["recommendations"].append("Replace gender-biased terms with gender-neutral alternatives")
        
    if supportive_term_count == 0 and inclusive_term_count == 0:
        report["recommendations"].append("Consider adding inclusive language to encourage diversity")
    
    return normalized_bias_score, report

def extract_job_keywords(text, limit=30):
    """
    Enhanced keyword extraction using NLP techniques, TF-IDF principles,
    and domain knowledge from job descriptions.
    """
    # Process with spaCy
    doc = nlp(text)
    
    # Container for keywords with scores
    keyword_scores = {}
    
    # Extract predefined skill keywords with context
    for skill in SKILL_KEYWORDS:
        skill_lower = skill.lower()
        if skill_lower in text.lower():
            # Check for context around skill
            contexts = re.findall(r'(\w+\s+){0,3}' + re.escape(skill_lower) + r'(\s+\w+){0,3}', text.lower())
            if contexts:
                # Give higher score to skills with related terms nearby
                context_score = 1.0
                for context in contexts:
                    context_text = ' '.join([c for c in context if c.strip()])
                    if any(related in context_text for related in ['experienced', 'proficient', 'skilled', 'knowledge']):
                        context_score += 0.5
                keyword_scores[skill] = context_score
            else:
                keyword_scores[skill] = 1.0
    
    # Extract industry-specific terms
    for industry, terms in INDUSTRY_TERMS.items():
        industry_matches = []
        for term in terms:
            if term.lower() in text.lower():
                industry_matches.append(term)
                keyword_scores[term] = 1.0
        
        # If multiple terms from same industry, add industry name too
        if len(industry_matches) >= 2:
            keyword_scores[industry] = 1.5
    
    # Extract entities with contextual filtering
    entity_types = ['ORG', 'PRODUCT', 'LANGUAGE', 'GPE', 'TECH', 'EVENT']
    for ent in doc.ents:
        if ent.label_ in entity_types:
            # Check if entity is in a relevant context
            sent = next((s for s in doc.sents if ent.start >= s.start and ent.end <= s.end), None)
            if sent:
                context = sent.text.lower()
                if any(term in context for term in ['experience', 'knowledge', 'skill', 'tool', 'technology', 'platform']):
                    keyword_scores[ent.text] = 1.25
                else:
                    keyword_scores[ent.text] = 0.75
    
    # Extract noun phrases and technical terms
    for chunk in doc.noun_chunks:
        # Filter for relevant technical and professional phrases
        if len(chunk.text.split()) >= 2:  # Multi-word phrases are more likely meaningful
            chunk_text = chunk.text.strip()
            
            # Check if it's a technical term or related to professional skills
            if any(skill_word in chunk_text.lower() for skill_word in ['software', 'programming', 'tool', 'system', 'analysis', 'management']):
                keyword_scores[chunk_text] = 1.2
            elif any(chunk.root.text.lower() == word for word in ['experience', 'skill', 'knowledge', 'ability', 'proficiency']):
                keyword_scores[chunk_text] = 1.1
    
    # Extract terms related to job roles and hierarchies
    for role_category, roles in ROLE_HIERARCHIES.items():
        for role in roles:
            if role.lower() in text.lower():
                keyword_scores[role] = 1.0
                # Add role category with lower score
                keyword_scores[role_category] = keyword_scores.get(role_category, 0) + 0.5
    
    # Find words/phrases that appear in important positions (headings, bullet points, etc.)
    heading_matches = re.findall(r'^[A-Z][A-Za-z\s]+:|^\s*•\s*([A-Za-z\s]+)', text, re.MULTILINE)
    for match in heading_matches:
        match_text = match.strip(': •').strip()
        if match_text and len(match_text.split()) <= 4:  # Reasonable length for a skill
            keyword_scores[match_text] = keyword_scores.get(match_text, 0) + 1.0
    
    # Use spaCy's word vectors to find terms similar to known skills
    if SKILL_KEYWORDS:
        sample_skills = [skill for skill in SKILL_KEYWORDS if skill in nlp.vocab][:5]
        if sample_skills:
            # Find similar words in the document
            for token in doc:
                if token.has_vector and token.is_alpha and len(token.text) > 3:
                    # Check similarity with known skills
                    avg_similarity = sum(token.similarity(nlp(skill)) for skill in sample_skills) / len(sample_skills)
                    if avg_similarity > 0.7:  # High similarity threshold
                        keyword_scores[token.text] = keyword_scores.get(token.text, 0) + avg_similarity
    
    # Sort keywords by score in descending order
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Remove duplicates and near-duplicates
    unique_keywords = []
    for keyword, score in sorted_keywords:
        # Skip if keyword is too similar to already included keywords
        if not any(nlp(keyword).similarity(nlp(included)) > 0.85 for included in unique_keywords):
            unique_keywords.append(keyword)
            if len(unique_keywords) >= limit:
                break
    
    # Return keywords with their scores
    return {keyword: keyword_scores[keyword] for keyword in unique_keywords}

def analyze_sentiment(text, detailed=False):
    """
    Enhanced sentiment analysis with aspect-based analysis and detailed scoring.
    """
    # Basic sentiment analysis using transformer model
    inputs = sentiment_tokenizer(text, return_tensors="pt", 
                               max_length=app.config.get('MAX_SEQUENCE_LENGTH', 512), 
                               truncation=True)
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probabilities[0][1].item()
    
    if not detailed:
        return positive_score
    
    # Perform aspect-based sentiment analysis
    aspects = {
        'company_culture': ['culture', 'environment', 'team', 'workplace', 'community'],
        'job_requirements': ['requirement', 'qualification', 'experience', 'skill', 'education'],
        'job_benefits': ['benefit', 'salary', 'compensation', 'bonus', 'insurance', 'vacation', 'remote'],
        'diversity_inclusion': ['diversity', 'inclusion', 'equal', 'opportunity', 'gender', 'minority']
    }
    
    doc = nlp(text)
    aspect_scores = {}
    
    # Analyze sentiment for each aspect
    for aspect_name, aspect_terms in aspects.items():
        aspect_mentions = []
        
        # Find sentences mentioning aspect terms
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(term in sent_text for term in aspect_terms):
                aspect_mentions.append(sent.text)
        
        # Calculate sentiment for the aspect if mentioned
        if aspect_mentions:
            combined_text = " ".join(aspect_mentions)
            inputs = sentiment_tokenizer(combined_text, return_tensors="pt", 
                                        max_length=app.config.get('MAX_SEQUENCE_LENGTH', 512), 
                                        truncation=True)
            
            with torch.no_grad():
                outputs = sentiment_model(**inputs)
            
            aspect_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            aspect_scores[aspect_name] = {
                'score': aspect_probabilities[0][1].item(),
                'mentions': len(aspect_mentions),
                'examples': aspect_mentions[:3]  # Include some example mentions
            }
    
    # Return detailed sentiment analysis
    return {
        'overall_sentiment': positive_score,
        'aspect_sentiments': aspect_scores
    }

def analyze_user_query(query):
    """
    Advanced query analysis using contextual embeddings and semantic similarity.
    """
    # Preprocess query
    processed_query = preprocess_text(query, remove_stopwords=False)
    
    # Get embedding for the query
    query_embedding = get_sentence_embedding(processed_query)
    
    # Define intent categories with example queries for each
    intent_examples = {
        'job_search': [
            "Find software developer jobs",
            "I'm looking for data analyst positions",
            "Show me marketing jobs in New York",
            "Search for remote engineering opportunities",
            "Help me find entry-level positions"
        ],
        'salary_info': [
            "What's the average salary for a product manager?",
            "How much do software engineers make?",
            "Tell me about compensation for marketing roles",
            "Salary expectations for data scientists",
            "What's the pay range for UX designers?"
        ],
        'company_info': [
            "Tell me about companies with good work-life balance",
            "Which organizations support women in tech?",
            "Information about family-friendly employers",
            "Companies with the best diversity practices",
            "Tell me about workplace culture at tech startups"
        ],
        'skill_development': [
            "How to improve my Python skills",
            "Courses for learning project management",
            "Resources for developing leadership abilities",
            "What skills should I learn for data science?",
            "Training programs for UX design"
        ],
        'resume_help': [
            "Help me improve my resume",
            "Tips for writing a cover letter",
            "How to highlight achievements on my CV",
            "Make my resume stand out for tech jobs",
            "Review my application materials"
        ],
        'interview_prep': [
            "Common interview questions for software engineers",
            "How to prepare for behavioral interviews",
            "Tips for technical interviews",
            "Practice questions for data science interviews",
            "What should I expect in a panel interview?"
        ],
        'discrimination': [
            "What to do about gender bias in my workplace",
            "Dealing with discrimination during interviews",
            "Laws about workplace harassment",
            "Reporting unfair treatment at work",
            "Resources for fighting workplace inequality"
        ],
        'career_advice': [
            "How to transition to a new career",
            "Should I take this job offer?",
            "Tips for negotiating a promotion",
            "Career path options for programmers",
            "When is the right time to switch jobs?"
        ],
        'networking': [
            "How to build my professional network",
            "Tips for LinkedIn networking",
            "Finding mentors in my industry",
            "Professional associations I should join",
            "Best practices for following up after networking events"
        ],
        'work_life_balance': [
            "Companies with good work-life balance",
            "How to maintain balance in a demanding job",
            "Tips for preventing burnout",
            "Jobs that offer flexibility for parents",
            "Negotiating flexible working arrangements"
        ]
    }
    
    # Calculate embeddings for all example queries
    intent_embeddings = {}
    for intent, examples in intent_examples.items():
        # Get embeddings for each example
        example_embeddings = [get_sentence_embedding(ex) for ex in examples]
        # Average the embeddings to get a representative embedding for the intent
        intent_embeddings[intent] = np.mean(example_embeddings, axis=0)
    
    # Calculate similarity between query and each intent
    similarities = {}
    for intent, embedding in intent_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities[intent] = similarity
    
    # Get top 2 intents
    sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    primary_intent = sorted_intents[0][0]
    primary_score = sorted_intents[0][1]
    
    # If the top score is low, we might be dealing with a mixed intent
    secondary_intent = None
    secondary_score = 0
    if len(sorted_intents) > 1:
        secondary_intent = sorted_intents[1][0]
        secondary_score = sorted_intents[1][1]
    
    # Check if we have a clear primary intent or mixed intents
    is_mixed_intent = (primary_score < 0.7) and (secondary_score > 0.5)
    
    # Extract parameters for intent fulfillment
    params = extract_query_parameters(query, primary_intent, secondary_intent if is_mixed_intent else None)
    
    # Prepare result
    result = {
        'primary_intent': primary_intent,
        'primary_score': float(primary_score),
        'params': params
    }
    
    if is_mixed_intent:
        result['secondary_intent'] = secondary_intent
        result['secondary_score'] = float(secondary_score)
        result['is_mixed_intent'] = True
    
    return result

def extract_query_parameters(query, primary_intent, secondary_intent=None):
    """
    Extract relevant parameters from the query based on identified intent.
    """
    # Process the query with spaCy
    doc = nlp(query)
    params = {}
    
    # Extract job titles/roles
    if primary_intent in ['job_search', 'salary_info', 'resume_help']:
        # Look for job titles
        job_titles = []
        job_title_patterns = [
            # Pattern 1: ADJ? NOUN+ "job/position/role/work"
            [(token.text, token.pos_) for token in span if token.pos_ in ['ADJ', 'NOUN']]
            for span in doc.noun_chunks 
            if any(token.lemma_ in ['job', 'position', 'role', 'work'] for token in span)
        ]
        
        # Pattern 2: "as a/an" + NOUN PHRASE
        as_a_matches = re.findall(r'as an?|looking for an?|find an?|seeking an?|want an?|become an?)\s+([A-Za-z\s]+)', query)
        
        # Process Pattern 1 results
        for token_pattern in job_title_patterns:
            if token_pattern:
                job_title = ' '.join([text for text, pos in token_pattern])
                if job_title and job_title.lower() not in ['job', 'position', 'role', 'work']:
                    job_titles.append(job_title)
        
        # Process Pattern 2 results
        for match in as_a_matches:
            title = match.strip()
            if title and len(title.split()) <= 5:  # Reasonable length check
                job_titles.append(title)
        
        # Look for job titles from ROLE_HIERARCHIES
        for role_category, roles in ROLE_HIERARCHIES.items():
            for role in roles:
                if role.lower() in query.lower():
                    job_titles.append(role)
                    break
        
        # Filter out duplicates and add to params
        if job_titles:
            # Remove duplicates and keep the longest version
            job_titles.sort(key=len, reverse=True)
            unique_titles = []
            for title in job_titles:
                if not any(title.lower() in existing.lower() or existing.lower() in title.lower() for existing in unique_titles):
                    unique_titles.append(title)
            
            params['job_title'] = unique_titles[0]
            if len(unique_titles) > 1:
                params['alternative_titles'] = unique_titles[1:3]  # Add alternative titles
    
    # Extract locations (for job search, etc.)
    if primary_intent in ['job_search', 'company_info']:
        locations = []
        
        # Extract GPE and LOC entities
        location_entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        
        # Look for preposition + location patterns
        for token in doc:
            if token.dep_ == 'pobj' and token.head.dep_ == 'prep' and token.head.text.lower() in ['in', 'at', 'near', 'around', 'from']:
                if token.ent_type_ in ['GPE', 'LOC']:
                    locations.append(token.text)
        
        # Add "remote" as a location if mentioned
        if 'remote' in query.lower():
            locations.append('Remote')
        
        if locations:
            params['location'] = locations[0]
            if len(locations) > 1:
                params['alternative_locations'] = locations[1:]
    
    # Extract skills (for skill development, resume help)
    if primary_intent in ['skill_development', 'resume_help', 'job_search']:
        skills = []
        
        # Look for skill keywords
        for skill in SKILL_KEYWORDS:
            if skill.lower() in query.lower():
                skills.append(skill)
        
        # Look for skill patterns
        skill_patterns = [
            r'learn\s+([A-Za-z\s]{2,30})',
            r'improve\s+(?:my|our)?\s*([A-Za-z\s]{2,30})',
            r'develop\s+(?:my|our)?\s*([A-Za-z\s]{2,30})',
            r'practice\s+(?:my|our)?\s*([A-Za-z\s]{2,30})',
            r'skills?\s+(?:in|for|with)\s+([A-Za-z\s]{2,30})'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                clean_skill = match.strip().rstrip('., ')
                if clean_skill and len(clean_skill.split()) <= 5:  # Reasonable length check
                    skills.append(clean_skill)
        
        if skills:
            params['skill_area'] = skills[0]
            if len(skills) > 1:
                params['alternative_skills'] = skills[1:3]
    
    # Extract experience level
    experience_patterns = [
        r'(entry[- ]level|junior|senior|mid[- ]level|beginner|intermediate|advanced|expert)',
        r'([0-9]+)[- ]years?[- ]experience',
        r'experience:?\s*([0-9]+)[- ]years?'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            params['experience_level'] = matches[0]
            break
    
    # Extract industry information
    if primary_intent in ['job_search', 'company_info', 'career_advice']:
        industries = []
        
        # Look for industry terms
        for industry, terms in INDUSTRY_TERMS.items():
            if any(term.lower() in query.lower() for term in terms):
                industries.append(industry)
        
        # Look for industry patterns
        industry_patterns = [
            r'in\s+the\s+([A-Za-z\s]{2,30})\s+industry',
            r'([A-Za-z\s]{2,30})\s+sector',
            r'([A-Za-z\s]{2,30})\s+field'
        ]
        
        for pattern in industry_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                clean_industry = match.strip().rstrip('., ')
                if clean_industry and len(clean_industry.split()) <= 3:  # Reasonable length check
                    industries.append(clean_industry)
        
        if industries:
            params['industry'] = industries[0]
            if len(industries) > 1:
                params['alternative_industries'] = industries[1:3]
    
    # Extract company information
    if primary_intent in ['company_info', 'salary_info']:
        companies = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        if companies:
            params['company'] = companies[0]
            if len(companies) > 1:
                params['alternative_companies'] = companies[1:3]
    
    # Extract time frame information
    time_patterns = [
        r'(this week|this month|this year|next week|next month|next year)',
        r'in\s+([0-9]+)\s+(days|weeks|months|years)',
        r'within\s+([0-9]+)\s+(days|weeks|months|years)'
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                params['time_frame'] = ' '.join(matches[0])
            else:
                params['time_frame'] = matches[0]
            break
    
    return params

def generate_response_template(intent_info, include_alternatives=True):
    """
    Generate personalized response template based on intent analysis.
    """
    if isinstance(intent_info, str):
        # Simple intent string provided
        intent = intent_info
        params = {}
        is_mixed_intent = False
        secondary_intent = None
    else:
        # Full intent analysis provided
        intent = intent_info.get('primary_intent')
        params = intent_info.get('params', {})
        is_mixed_intent = intent_info.get('is_mixed_intent', False)
        secondary_intent = intent_info.get('secondary_intent')
    
    # Initialize response components
    greeting = "I understand you're interested in "
    main_response = ""
    followup_questions = []
    
    # Generate response based on intent
    if intent == 'job_search':
        job_title = params.get('job_title', 'roles')
        location = params.get('location', 'various locations')
        
        greeting += f"finding {job_title} positions"
        if location != 'various locations':
            greeting += f" in {location}"
        greeting += "."
        
        main_response = f"Here are some {job_title} opportunities that match your criteria. "
        
        if include_alternatives and 'alternative_titles' in params:
            main_response += f"I've also included some similar roles like {', '.join(params['alternative_titles'])} that might interest you. "
        
        followup_questions = [
            "Would you like to filter these results by experience level?",
            "Are you looking for remote positions or specific locations?",
            "Would you like information about typical salaries for these roles?"
        ]
        
    elif intent == 'salary_info':
        job_title = params.get('job_title', 'professionals in this field')
        
        greeting += f"salary information for {job_title}."
        
        main_response = f"Based on current market data, {job_title} typically earn "
        main_response += "the following compensation ranges, which can vary based on location, experience, and specific employer. "
        
        followup_questions = [
            "Would you like salary information for a specific location?",
            "Are you interested in compensation trends over time?",
            "Would you like to know about negotiation strategies for this role?"
        ]
        
    elif intent == 'company_info':
        company = params.get('company', 'companies')
        industry = params.get('industry', 'this industry')
        
        if company != 'companies':
            greeting += f"learning about {company}."
            main_response = f"Here's some key information about {company}, including culture, working environment, and diversity practices. "
        else:
            greeting += f"companies in {industry}."
            main_response = f"Here are some notable organizations in {industry} known for their workplace culture and practices. "
        
        followup_questions = [
            "Would you like to know about specific diversity initiatives?",
            "Are you interested in employee reviews or satisfaction ratings?",
            "Would you like to compare multiple companies side by side?"
        ]
        
    elif intent == 'skill_development':
        skill_area = params.get('skill_area', 'professional skills')
        
        greeting += f"developing your {skill_area} abilities."
        
        main_response = f"Here are some effective resources and approaches for enhancing your {skill_area} skills. "
        
        followup_questions = [
            "What's your current level of experience with this skill?",
            "Are you looking for free resources or paid courses?",
            "Would you like industry-specific applications of this skill?"
        ]
        
    elif intent == 'resume_help':
        job_title = params.get('job_title', 'your target role')
        
        greeting += "improving your resume"
        if job_title != 'your target role':
            greeting += f" for {job_title} positions"
        greeting += "."
        
        main_response = "Here are tailored recommendations to strengthen your resume and highlight your relevant qualifications. "
        
        followup_questions = [
            "Would you like specific advice on formatting and structure?",
            "Are there specific achievements you're trying to emphasize?",
            "Would you like examples of effective resume language for your field?"
        ]
        
    elif intent == 'interview_prep':
        job_title = params.get('job_title', 'your upcoming interview')
        
        greeting += f"preparing for {job_title} interviews."
        
        main_response = "Here are key interview strategies and common questions to help you prepare effectively. "
        
        followup_questions = [
            "Would you like to focus on technical or behavioral questions?",
            "Are you preparing for a specific interview format?",
            "Would you like advice on following up after the interview?"
        ]
        
    elif intent == 'discrimination':
        greeting += "addressing workplace discrimination concerns."
        
        main_response = "Here are resources and approaches for handling discrimination or bias in professional settings. "
        
        followup_questions = [
            "Are you looking for legal information or practical advice?",
            "Would you like resources specific to your situation?",
            "Are you interested in organization-level policies or individual responses?"
        ]
        
    elif intent == 'career_advice':
        greeting += "career guidance and professional development."
        
        main_response = "Here are some insights and considerations to help with your career decisions. "
        
        followup_questions = [
            "Where are you currently in your career journey?",
            "What are your primary career goals at this stage?",
            "Would you like advice on a specific career decision?"
        ]
        
    elif intent == 'networking':
        greeting += "building your professional network."
        
        main_response = "Here are effective networking strategies to expand your professional connections. "
        
        followup_questions = [
            "Are you looking to network within a specific industry?",
            "Would you like online or in-person networking strategies?",
            "Are you interested in finding mentors or peers in your field?"
        ]
        
    elif intent == 'work_life_balance':
        greeting += "achieving better work-life balance."
        
        main_response = "Here are approaches and options for maintaining balance between your professional and personal life. "
        
        followup_questions = [
            "Are you dealing with a specific work-life challenge?",
            "Would you like information about companies with strong work-life policies?",
            "Are you interested in flexible working arrangements?"
        ]
        
    else:
        # Generic response for unclassified intents
        greeting = "I understand your query. "
        main_response = "Here's information that might help address your needs. "
        followup_questions = [
            "Could you tell me more about what you're looking for?",
            "Would you like more specific information on any aspect of this topic?",
            "Is there another way I can help with your query?"
        ]
    
    # Handle mixed intents
    if is_mixed_intent and secondary_intent:
        main_response += f"\n\nI also notice you're interested in aspects related to {secondary_intent.replace('_', ' ')}. "
        
        if secondary_intent == 'salary_info':
            main_response += "I can provide compensation information for these roles as well. "
        elif secondary_intent == 'company_info':
            main_response += "I can include details about relevant companies in this field. "
        elif secondary_intent == 'skill_development':
            main_response += "I can suggest skill development resources that would enhance your prospects. "
    
    # Compile the final response template
    response_template = {
        'greeting': greeting,
        'main_response': main_response,
        'followup_questions': followup_questions[:2] if followup_questions else []  # Limit to 2 follow-up questions
    }
    
    return response_template

def get_response_for_query(query, detailed=False):
    """
    Process a user query and generate a comprehensive response.
    """
    # Analyze the user's query intent
    intent_analysis = analyze_user_query(query)
    
    # Generate a personalized response template
    response_template = generate_response_template(intent_analysis)
    
    # Prepare the response components
    greeting = response_template['greeting']
    main_response = response_template['main_response']
    followup_questions = response_template['followup_questions']
    
    # Build the final response
    response = f"{greeting}\n\n{main_response}"
    
    # Add follow-up questions if appropriate
    if followup_questions:
        response += "\n\nTo better assist you, I'd like to know:\n"
        for question in followup_questions:
            response += f"- {question}\n"
    
    if detailed:
        # Include the intent analysis for debugging or advanced applications
        return {
            'response': response,
            'intent_analysis': intent_analysis,
            'template': response_template
        }
    
    return response

def get_job_posting_analysis(job_text):
    """
    Comprehensive analysis of a job posting including bias detection,
    keyword extraction, and sentiment analysis.
    """
    # Check if text is empty or too short
    if not job_text or len(job_text.strip()) < 50:
        return {
            'error': 'Text too short or empty',
            'message': 'Please provide a complete job posting for analysis.'
        }
    
    # Run analysis components
    bias_score, bias_details = detect_gender_bias(job_text, detailed=True)
    keywords = extract_job_keywords(job_text)
    sentiment_analysis = analyze_sentiment(job_text, detailed=True)
    
    # Generate overall assessment
    assessment = "This job posting "
    
    # Assess bias
    if bias_score < 0.3:
        assessment += "uses inclusive language and shows minimal gender bias. "
    elif bias_score < 0.6:
        assessment += "contains some gendered language that could be more inclusive. "
    else:
        assessment += "contains significant gender-coded language that may discourage diverse applicants. "
    
    # Assess keywords
    if keywords:
        top_keywords = list(keywords.keys())[:5]
        assessment += f"Emphasizes skills in {', '.join(top_keywords)}. "
    
    # Assess sentiment
    overall_sentiment = sentiment_analysis['overall_sentiment']
    if overall_sentiment > 0.7:
        assessment += "The tone is very positive and welcoming. "
    elif overall_sentiment > 0.5:
        assessment += "The tone is generally positive. "
    else:
        assessment += "The tone is neutral or could be more engaging. "
    
    # Check for aspect-specific sentiment
    aspects = sentiment_analysis.get('aspect_sentiments', {})
    
    if 'diversity_inclusion' in aspects:
        di_score = aspects['diversity_inclusion']['score']
        if di_score > 0.7:
            assessment += "It strongly emphasizes diversity and inclusion. "
        elif di_score > 0.5:
            assessment += "It mentions diversity and inclusion positively. "
    else:
        assessment += "It could benefit from more explicit diversity and inclusion language. "
    
    # Add recommendations
    recommendations = []
    
    # Bias recommendations
    if bias_score >= 0.3:
        if bias_details['masculine_terms']:
            top_masculine = sorted(bias_details['masculine_terms'].items(), key=lambda x: x[1], reverse=True)[:3]
            terms = [term for term, _ in top_masculine]
            recommendations.append(f"Consider replacing masculine-coded terms like '{', '.join(terms)}' with more neutral alternatives.")
        
        if bias_details['biased_terms']:
            top_biased = sorted(bias_details['biased_terms'].items(), key=lambda x: x[1], reverse=True)[:3]
            terms = [term for term, _ in top_biased]
            recommendations.append(f"Replace gender-specific terms like '{', '.join(terms)}' with inclusive language.")
        
        if not bias_details['inclusive_terms']:
            recommendations.append("Add inclusive language that explicitly welcomes diverse candidates.")
    
    # Keyword recommendations
    if len(keywords) < 10:
        recommendations.append("Include more specific skills and qualifications to attract qualified candidates.")
    
    # Sentiment recommendations
    if overall_sentiment < 0.5:
        recommendations.append("Use more positive language to create an engaging and welcoming tone.")
    
    aspect_recommendations = {
        'company_culture': "Add more details about company culture and work environment.",
        'job_benefits': "Include specific benefits and perks to make the position more attractive.",
        'diversity_inclusion': "Explicitly state commitment to diversity and equal opportunity."
    }
    
    for aspect, recommendation in aspect_recommendations.items():
        if aspect not in aspects:
            recommendations.append(recommendation)
    
    # Compile final analysis
    analysis_result = {
        'assessment': assessment,
        'recommendations': recommendations,
        'bias': {
            'score': bias_score,
            'details': bias_details
        },
        'keywords': keywords,
        'sentiment': sentiment_analysis
    }
    
    return analysis_result

# Additional utility functions to support the module

def compare_job_postings(job_text1, job_text2):
    """
    Compare two job postings for similarities, differences, and relative inclusivity.
    """
    # Analyze both postings
    analysis1 = get_job_posting_analysis(job_text1)
    analysis2 = get_job_posting_analysis(job_text2)
    
    # Calculate text similarity
    text1_embedding = get_sentence_embedding(preprocess_text(job_text1))
    text2_embedding = get_sentence_embedding(preprocess_text(job_text2))
    similarity = float(cosine_similarity([text1_embedding], [text2_embedding])[0][0])
    
    # Compare bias scores
    bias_diff = analysis1['bias']['score'] - analysis2['bias']['score']
    
    # Compare keywords
    keywords1 = set(analysis1['keywords'].keys())
    keywords2 = set(analysis2['keywords'].keys())
    common_keywords = keywords1.intersection(keywords2)
    unique_keywords1 = keywords1 - keywords2
    unique_keywords2 = keywords2 - keywords1
    
    # Compare sentiment
    sentiment_diff = analysis1['sentiment']['overall_sentiment'] - analysis2['sentiment']['overall_sentiment']
    
    # Generate comparison summary
    comparison = {
        'similarity': similarity,
        'bias_comparison': {
            'difference': bias_diff,
            'more_inclusive': 1 if bias_diff < 0 else 2 if bias_diff > 0 else 'equal'
        },
        'keywords': {
            'common': list(common_keywords),
            'unique_to_first': list(unique_keywords1),
            'unique_to_second': list(unique_keywords2)
        },
        'sentiment_comparison': {
            'difference': sentiment_diff,
            'more_positive': 1 if sentiment_diff > 0 else 2 if sentiment_diff < 0 else 'equal'
        },
        'recommendations': []
    }
    
    # Generate comparative recommendations
    if bias_diff > 0.2:
        comparison['recommendations'].append("The first posting uses more gendered language. Consider adopting more inclusive language from the second posting.")
    elif bias_diff < -0.2:
        comparison['recommendations'].append("The second posting uses more gendered language. Consider adopting more inclusive language from the first posting.")
    
    if len(unique_keywords1) > len(unique_keywords2):
        comparison['recommendations'].append("The first posting mentions more unique skills and qualifications. Consider incorporating relevant ones into the second posting.")
    elif len(unique_keywords2) > len(unique_keywords1):
        comparison['recommendations'].append("The second posting mentions more unique skills and qualifications. Consider incorporating relevant ones into the first posting.")
    
    if sentiment_diff > 0.2:
        comparison['recommendations'].append("The first posting has a more positive tone. Consider making the second posting more engaging.")
    elif sentiment_diff < -0.2:
        comparison['recommendations'].append("The second posting has a more positive tone. Consider making the first posting more engaging.")
    
    return comparison

def suggest_inclusive_alternatives(text):
    """
    Suggest more inclusive alternatives for potentially biased language in text.
    """
    doc = nlp(text)
    suggestions = []
    
    # Check for biased terms
    for word in GENDER_BIASED_WORDS:
        if word in text.lower():
            # Find appropriate alternatives
            if word in job_data.get('bias_alternatives', {}):
                alternatives = job_data['bias_alternatives'][word]
                suggestions.append({
                    'original': word,
                    'alternatives': alternatives,
                    'context': f"...{text[max(0, text.lower().find(word) - 30):text.lower().find(word) + len(word) + 30]}..."
                })
    
    # Check for masculine-coded words
    for word in MASCULINE_CODED_WORDS:
        if word in text.lower():
            # Find sentences with this word
            for sent in doc.sents:
                if word in sent.text.lower():
                    # Suggest neutral alternatives
                    neutral_alternatives = job_data.get('neutral_alternatives', {}).get(word, [])
                    if neutral_alternatives:
                        suggestions.append({
                            'original': word,
                            'alternatives': neutral_alternatives,
                            'context': sent.text
                        })
    
    # Check for common phrases with bias
    biased_phrases = [
        ('he or she', 'they'),
        ('him or her', 'them'),
        ('his or her', 'their'),
        ('man hours', 'work hours'),
        ('manpower', 'workforce'),
        ('manning', 'staffing'),
        ('chairman', 'chairperson'),
        ('guys', 'team/folks/everyone'),
        ('manmade', 'artificial/synthetic'),
        ('freshman', 'first-year student'),
        ('policeman', 'police officer'),
        ('fireman', 'firefighter')
    ]
    
    for phrase, alternative in biased_phrases:
        if phrase in text.lower():
            suggestions.append({
                'original': phrase,
                'alternatives': [alternative],
                'context': f"...{text[max(0, text.lower().find(phrase) - 30):text.lower().find(phrase) + len(phrase) + 30]}..."
            })
    
    return suggestions