import re
import json
from datetime import datetime
from app.ml_models.job_matcher import RealTimeJobMatcher
from app.ml_models.resume_optimizer import ResumeOptimizer
from app.ml_models.salary_predictor import SalaryPredictor, SalaryDataCollector, MarketAnalyzer

class ChatService:
    def __init__(self):
        self.job_matcher = RealTimeJobMatcher()
        self.resume_optimizer = ResumeOptimizer()
        
        # Initialize salary predictor with required dependencies
        self.data_collector = SalaryDataCollector()
        self.market_analyzer = MarketAnalyzer(self.data_collector)
        self.salary_predictor = SalaryPredictor(self.data_collector, self.market_analyzer)
        
        # Indian job market specific intents and responses
        self.intent_patterns = {
            'job_search': [
                r'find.*job', r'job.*search', r'looking.*job', r'job.*opening',
                r'vacancy', r'position.*available', r'hiring', r'recruitment'
            ],
            'salary': [
                r'salary', r'pay', r'compensation', r'wage', r'ctc', r'package',
                r'negotiate.*salary', r'salary.*range', r'how much.*earn'
            ],
            'resume': [
                r'resume', r'cv', r'curriculum.*vitae', r'resume.*build',
                r'resume.*improve', r'resume.*tip', r'ats.*score'
            ],
            'interview': [
                r'interview', r'interview.*prep', r'interview.*question',
                r'technical.*round', r'hr.*round', r'coding.*interview'
            ],
            'company': [
                r'company.*culture', r'work.*life.*balance', r'company.*review',
                r'best.*company', r'woman.*friendly', r'maternity.*leave'
            ],
            'career_growth': [
                r'career.*growth', r'promotion', r'skill.*development',
                r'upskill', r'career.*change', r'switch.*company'
            ],
            'location': [
                r'bangalore', r'mumbai', r'delhi', r'hyderabad', r'chennai',
                r'pune', r'gurgaon', r'noida', r'remote.*work'
            ]
        }
        
        self.indian_context_responses = {
            'job_search_tips': [
                "For effective job search in India, focus on these platforms: Naukri.com, LinkedIn India, AngelList, Glassdoor India, and company career pages.",
                "Update your LinkedIn profile with Indian keywords and location preferences. Many recruiters actively search LinkedIn for candidates.",
                "Consider reaching out to alumni from your college who work in your target companies. The alumni network is very strong in India.",
                "Apply through employee referrals when possible - many Indian companies have strong referral programs.",
                "Don't forget about job fairs and campus placements if you're a recent graduate."
            ],
            'salary_negotiation_india': [
                "In India, salary negotiation often includes base salary, variable pay, ESOPs (for startups), PF contribution, and benefits.",
                "Research salary ranges on AmbitionBox, Glassdoor India, and PayScale for Indian market rates.",
                "Consider the total CTC (Cost to Company) including benefits like health insurance, meals, transport allowance.",
                "Startups might offer lower base salary but higher equity. Evaluate the company's growth potential.",
                "Notice period in India is typically 1-3 months. Factor this into your negotiation timeline."
            ],
            'women_specific_advice': [
                "Many Indian companies now have women-friendly policies. Ask about maternity leave (minimum 26 weeks in India), childcare support, and flexible working hours.",
                "Look for companies with women leadership programs and diversity initiatives.",
                "Consider joining women in tech groups like Women Who Code Delhi, PyLadies India, or company-specific women groups.",
                "Some companies offer special benefits like cab services for late working hours and women safety programs.",
                "Network with other women professionals through LinkedIn groups focused on Indian women in tech."
            ]
        }

    def process_message(self, message, user_id=None, context=None):
        """Process user message and generate appropriate response"""
        message_lower = message.lower()
        intent = self._detect_intent(message_lower)
        
        response_data = {
            'message': '',
            'intent': intent,
            'suggestions': [],
            'action_required': None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if intent == 'job_search':
            response_data['message'] = self._handle_job_search(message_lower, context)
        elif intent == 'salary':
            response_data['message'] = self._handle_salary_query(message_lower, context)
        elif intent == 'resume':
            response_data['message'] = self._handle_resume_query(message_lower, context)
        elif intent == 'interview':
            response_data['message'] = self._handle_interview_query(message_lower, context)
        elif intent == 'company':
            response_data['message'] = self._handle_company_query(message_lower, context)
        elif intent == 'career_growth':
            response_data['message'] = self._handle_career_growth_query(message_lower, context)
        else:
            response_data['message'] = self._handle_general_query(message_lower, context)
            
        response_data['suggestions'] = self.get_follow_up_suggestions(message)
        
        return response_data

    def _detect_intent(self, message):
        """Detect user intent from message"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    return intent
        return 'general'

    def _handle_job_search(self, message, context):
        """Handle job search related queries"""
        location = self._extract_location(message)
        role = self._extract_role(message)
        
        response = "I can help you find job opportunities! "
        
        if location:
            response += f"For {location}, here are some tips:\n\n"
            if location in ['bangalore', 'bengaluru']:
                response += "Bangalore is India's Silicon Valley with opportunities in startups, product companies, and R&D centers of global firms."
            elif location in ['mumbai']:
                response += "Mumbai offers fintech, media, and corporate opportunities along with many multinational headquarters."
            elif location in ['delhi', 'gurgaon', 'noida']:
                response += "Delhi NCR has a mix of startups, consulting firms, and government tech initiatives."
            elif location in ['hyderabad']:
                response += "Hyderabad is strong in biotech, pharma IT, and has many global development centers."
            elif location in ['pune']:
                response += "Pune offers automotive tech, IT services, and has a growing startup ecosystem."
                
        response += "\n\nFor an effective job search in India, I recommend:\n"
        response += "1. Update your Naukri.com and LinkedIn profiles\n"
        response += "2. Set up job alerts for your target roles\n"
        response += "3. Leverage your college alumni network\n"
        response += "4. Consider employee referrals - they're very effective in India\n"
        response += "5. Research company culture and women-friendly policies\n\n"
        response += "Would you like me to suggest some companies or specific job openings based on your profile?"
        
        return response

    def _handle_salary_query(self, message, context):
        """Handle salary and compensation queries"""
        response = "Salary negotiation in India involves several components beyond just base salary:\n\n"
        response += "**Salary Components in India:**\n"
        response += "• Base Salary (60-70% of CTC)\n"
        response += "• Variable Pay/Bonus (10-20%)\n"
        response += "• Employee Provident Fund (EPF)\n"
        response += "• Health Insurance\n"
        response += "• ESOPs (Employee Stock Options) for startups\n"
        response += "• Other benefits: Meal vouchers, Transport allowance, etc.\n\n"
        
        response += "**Negotiation Tips for Indian Market:**\n"
        response += "1. Research on AmbitionBox, Glassdoor India for market rates\n"
        response += "2. Consider the total package, not just base salary\n"
        response += "3. Evaluate startup equity potential vs. established company stability\n"
        response += "4. Factor in notice period (usually 1-3 months in India)\n"
        response += "5. Ask about annual appraisal cycles and promotion timelines\n\n"
        
        response += "**For Women Specifically:**\n"
        response += "• Ensure 26+ weeks maternity leave (legal minimum in India)\n"
        response += "• Ask about return-to-work programs\n"
        response += "• Inquire about flexible working arrangements\n"
        response += "• Check for women safety programs and late-night transportation\n\n"
        
        response += "Would you like me to help you research salary ranges for a specific role and location?"
        
        return response

    def _handle_resume_query(self, message, context):
        """Handle resume building and optimization queries"""
        response = "Creating an effective resume for the Indian job market:\n\n"
        response += "**ATS-Friendly Format:**\n"
        response += "• Use standard fonts (Arial, Calibri) and simple formatting\n"
        response += "• Include keywords from job descriptions\n"
        response += "• Keep it 1-2 pages maximum\n"
        response += "• Save as both PDF and Word formats\n\n"
        
        response += "**Indian Resume Specifics:**\n"
        response += "• Include your photo (common in India, unlike Western countries)\n"
        response += "• Mention your current CTC and expected CTC\n"
        response += "• Include notice period information\n"
        response += "• Add relevant certifications (especially for tech roles)\n"
        response += "• Mention college CGPA if you're a recent graduate\n\n"
        
        response += "**Key Sections:**\n"
        response += "1. Professional Summary (2-3 lines)\n"
        response += "2. Technical Skills (very important for tech roles)\n"
        response += "3. Work Experience (with quantified achievements)\n"
        response += "4. Education (include percentage/CGPA)\n"
        response += "5. Projects (especially for software roles)\n"
        response += "6. Certifications and Awards\n\n"
        
        response += "**Common Mistakes to Avoid:**\n"
        response += "• Generic objectives instead of specific value propositions\n"
        response += "• Listing duties instead of achievements\n"
        response += "• Not tailoring resume for each application\n"
        response += "• Ignoring ATS optimization\n\n"
        
        response += "Would you like me to review your resume or help you build one from scratch?"
        
        return response

    def _handle_interview_query(self, message, context):
        """Handle interview preparation queries"""
        response = "Interview preparation for Indian companies:\n\n"
        response += "**Technical Interview Rounds:**\n"
        response += "• Coding problems (data structures, algorithms)\n"
        response += "• System design (for senior roles)\n"
        response += "• Technology deep-dive questions\n"
        response += "• Live coding sessions\n\n"
        
        response += "**HR/Behavioral Round:**\n"
        response += "• Why do you want to join our company?\n"
        response += "• Where do you see yourself in 5 years?\n"
        response += "• How do you handle work pressure?\n"
        response += "• Questions about notice period and salary expectations\n"
        response += "• Cultural fit assessment\n\n"
        
        response += "**Indian Company Culture Questions:**\n"
        response += "• Ability to work in diverse, multi-cultural teams\n"
        response += "• Flexibility with working hours (considering global clients)\n"
        response += "• Comfort with hierarchical structures\n"
        response += "• Experience with client interaction (for service companies)\n\n"
        
        response += "**Questions to Ask Them:**\n"
        response += "• Team structure and reporting hierarchy\n"
        response += "• Growth opportunities and career path\n"
        response += "• Learning and development programs\n"
        response += "• Work-life balance initiatives\n"
        response += "• Women-friendly policies (if relevant)\n\n"
        
        response += "**Specific Tips for Women:**\n"
        response += "• Research the company's diversity statistics\n"
        response += "• Ask about women leadership programs\n"
        response += "• Inquire about maternity policies and support\n"
        response += "• Check for women employee resource groups\n\n"
        
        response += "Would you like specific interview tips for any particular company or role?"
        
        return response

    def _handle_company_query(self, message, context):
        """Handle company culture and work environment queries"""
        response = "Here are some companies in India known for supporting women and maintaining good work culture:\n\n"
        response += "**Tech Product Companies:**\n"
        response += "• Flipkart: Strong maternity benefits, flexible working\n"
        response += "• Swiggy: Extended parental leave, return-to-work programs\n"
        response += "• Zomato: Gender neutral policies, menstrual leave\n"
        response += "• Paytm: Equal pay initiatives, women leadership programs\n"
        response += "• Razorpay: Flexible hours, comprehensive health coverage\n\n"
        
        response += "**Global Tech Companies (Indian Offices):**\n"
        response += "• Google India: Excellent work-life balance, diversity programs\n"
        response += "• Microsoft India: Strong inclusion initiatives\n"
        response += "• Amazon India: Comprehensive benefits, career development\n"
        response += "• Adobe India: Creative work environment, flexible policies\n\n"
        
        response += "**What to Look for in Women-Friendly Companies:**\n"
        response += "• 26+ weeks maternity leave (legal minimum)\n"
        response += "• Childcare support or daycare facilities\n"
        response += "• Flexible working hours and remote work options\n"
        response += "• Women safety programs (cab services, security)\n"
        response += "• Equal pay and promotion opportunities\n"
        response += "• Women leadership representation\n"
        response += "• Anti-harassment policies and committees\n\n"
        
        response += "**Red Flags to Avoid:**\n"
        response += "• Unclear or discriminatory hiring practices\n"
        response += "• No women in leadership positions\n"
        response += "• Poor reviews on platforms like Glassdoor India\n"
        response += "• Excessive working hours without compensation\n"
        response += "• No clear career progression path\n\n"
        
        response += "Would you like me to provide more details about any specific company or help you research a particular organization?"
        
        return response

    def _handle_career_growth_query(self, message, context):
        """Handle career development and growth queries"""
        response = "Career growth strategies for women in the Indian tech industry:\n\n"
        response += "**Skill Development:**\n"
        response += "• Stay updated with latest technologies and frameworks\n"
        response += "• Pursue relevant certifications (AWS, Google Cloud, etc.)\n"
        response += "• Contribute to open source projects\n"
        response += "• Build a strong portfolio showcasing your work\n\n"
        
        response += "**Networking in India:**\n"
        response += "• Join women in tech groups (Women Who Code, PyLadies India)\n"
        response += "• Attend meetups and conferences in your city\n"
        response += "• Leverage college alumni networks\n"
        response += "• Participate in company-sponsored tech talks and events\n"
        response += "• Engage actively on LinkedIn with industry content\n\n"
        
        response += "**Career Progression Paths:**\n"
        response += "• Individual Contributor → Senior IC → Staff/Principal Engineer\n"
        response += "• IC → Team Lead → Engineering Manager → Director\n"
        response += "• Technical → Product Management → Product Leadership\n"
        response += "• Corporate → Startup → Entrepreneurship\n\n"
        
        response += "**Switching Companies Strategically:**\n"
        response += "• Service company → Product company for better learning\n"
        response += "• Large corporate → Startup for more responsibility\n"
        response += "• Indian company → Global company for exposure\n"
        response += "• Aim for 20-40% salary increase with each switch\n\n"
        
        response += "**Building Personal Brand:**\n"
        response += "• Write technical blogs and articles\n"
        response += "• Speak at conferences and meetups\n"
        response += "• Mentor junior developers\n"
        response += "• Participate in hackathons and coding competitions\n\n"
        
        response += "**Overcoming Challenges as a Woman:**\n"
        response += "• Build confidence through continuous learning\n"
        response += "• Find mentors and sponsors within your organization\n"
        response += "• Don't hesitate to negotiate and advocate for yourself\n"
        response += "• Join women support groups for guidance and networking\n\n"
        
        response += "What specific aspect of career growth would you like to discuss further?"
        
        return response

    def _handle_general_query(self, message, context):
        """Handle general queries"""
        response = "I'm here to help you navigate your career journey in India! I can assist with:\n\n"
        response += "🔍 **Job Search**: Finding opportunities, company research, application strategies\n"
        response += "💰 **Salary & Negotiation**: Market rates, compensation analysis, negotiation tips\n"
        response += "📄 **Resume Building**: ATS optimization, Indian market specifics, review and feedback\n"
        response += "🎯 **Interview Prep**: Technical rounds, HR questions, company-specific guidance\n"
        response += "🏢 **Company Culture**: Women-friendly workplaces, benefits analysis\n"
        response += "📈 **Career Growth**: Skill development, networking, career transitions\n\n"
        
        response += "**Popular topics for women in Indian tech:**\n"
        response += "• Best cities for tech careers (Bangalore, Mumbai, Hyderabad)\n"
        response += "• Maternity and childcare policies\n"
        response += "• Work-life balance in different companies\n"
        response += "• Breaking into product management or leadership roles\n"
        response += "• Transitioning from service companies to product companies\n\n"
        
        response += "What would you like to know more about? Feel free to ask specific questions!"
        
        return response

    def _extract_location(self, message):
        """Extract location from message"""
        locations = {
            'bangalore': ['bangalore', 'bengaluru', 'blr'],
            'mumbai': ['mumbai', 'bombay'],
            'delhi': ['delhi', 'new delhi', 'ncr', 'gurgaon', 'gurugram', 'noida'],
            'hyderabad': ['hyderabad', 'hyd'],
            'chennai': ['chennai', 'madras'],
            'pune': ['pune'],
            'kolkata': ['kolkata', 'calcutta'],
            'ahmedabad': ['ahmedabad'],
            'kochi': ['kochi', 'cochin']
        }
        
        for city, variations in locations.items():
            for variation in variations:
                if variation in message:
                    return city
        return None

    def _extract_role(self, message):
        """Extract job role from message"""
        roles = [
            'software engineer', 'developer', 'programmer', 'sde',
            'data scientist', 'data analyst', 'ml engineer',
            'product manager', 'pm', 'product owner',
            'designer', 'ui designer', 'ux designer',
            'devops', 'sre', 'system admin',
            'qa', 'tester', 'quality assurance',
            'tech lead', 'engineering manager',
            'consultant', 'business analyst'
        ]
        
        for role in roles:
            if role in message:
                return role
        return None

    def get_follow_up_suggestions(self, message):
        """Generate follow-up suggestions based on user message"""
        message_lower = message.lower()
        
        if 'job' in message_lower:
            return [
                "Show me companies with good work-life balance",
                "What's the average salary for my role?",
                "Help me optimize my resume for ATS",
                "Interview tips for Indian tech companies"
            ]
        elif 'salary' in message_lower:
            return [
                "How to negotiate salary in India?",
                "Compare salaries across different cities",
                "What benefits should I ask for?",
                "Salary trends in my industry"
            ]
        elif 'resume' in message_lower:
            return [
                "Review my resume for ATS compatibility",
                "Show me resume templates",
                "How to highlight achievements?",
                "Keywords for my industry"
            ]
        elif 'interview' in message_lower:
            return [
                "Common technical interview questions",
                "How to research a company before interview?",
                "Behavioral interview preparation",
                "Questions to ask the interviewer"
            ]
        else:
            return [
                "Find software engineer jobs in Bangalore",
                "How to prepare for technical interviews?",
                "Companies with good maternity policies",
                "Career growth tips for women in tech"
            ]

    def get_location_comparison(self, role, experience):
        """Compare salaries across different Indian cities"""
        locations = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'chennai', 'pune']
        comparison = {}
        
        for location in locations:
            data = self.get_market_data(role, location, experience)
            comparison[location] = {
                'median_salary': data['salary_range']['median'],
                'cost_of_living_index': self._get_cost_of_living_index(location),
                'adjusted_salary': self._adjust_for_cost_of_living(
                    data['salary_range']['median'], 
                    location
                )
            }
        
        return comparison

    def get_growth_trends(self, role, location):
        """Get salary growth trends (simulated data)"""
        return {
            'yoy_growth': 12.5,  # Year over year growth percentage
            'five_year_projection': {
                '2025': 1500000,
                '2026': 1650000,
                '2027': 1815000,
                '2028': 1995000,
                '2029': 2195000
            },
            'market_trends': [
                "Remote work increasing salary standardization across tier-1 cities",
                "AI/ML roles seeing 15-20% higher growth than traditional software roles",
                "Product companies offering 20-30% premium over service companies",
                "Equity compensation becoming standard in startups"
            ]
        }

    def _get_experience_range(self, experience_years):
        """Convert experience years to range categories"""
        if experience_years <= 2:
            return '0-2'
        elif experience_years <= 5:
            return '2-5'
        elif experience_years <= 8:
            return '5-8'
        else:
            return '8+'

    def _get_cost_of_living_index(self, location):
        """Get cost of living index for Indian cities (Bangalore = 100)"""
        col_index = {
            'bangalore': 100,
            'mumbai': 120,
            'delhi': 110,
            'hyderabad': 85,
            'chennai': 90,
            'pune': 95,
            'kolkata': 80,
            'ahmedabad': 75
        }
        return col_index.get(location.lower(), 100)

    def _adjust_for_cost_of_living(self, salary, location):
        """Adjust salary for cost of living"""
        col_index = self._get_cost_of_living_index(location)
        return int(salary * (100 / col_index))

    def _get_market_position(self, median_salary, multiplier):
        """Determine market position based on salary"""
        adjusted_median = median_salary * multiplier
        if adjusted_median > 3000000:
            return "Premium"
        elif adjusted_median > 1500000:
            return "Above Market"
        elif adjusted_median > 800000:
            return "Market Rate"
        else:
            return "Below Market"

    def _get_default_market_data(self):
        """Return default market data when specific data is not available"""
        return {
            'role': 'Software Engineer',
            'location': 'Bangalore',
            'experience_range': '2-5',
            'salary_range': {
                'min': 800000,
                'median': 1500000,
                'max': 2500000
            },
            'percentile_25': 1000000,
            'percentile_50': 1500000,
            'percentile_75': 2000000,
            'percentile_90': 2500000,
            'currency': 'INR',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }