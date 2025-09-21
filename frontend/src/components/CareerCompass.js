import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  FileText, 
  DollarSign, 
  Users, 
  MessageCircle, 
  Lightbulb,
  Star,
  Send,
  Menu,
  X,
  Home,
  User,
  Briefcase,
  BookOpen,
  LogIn,
  UserPlus
} from 'lucide-react';

const CareerCompass = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([
    {
      type: 'assistant',
      message: "Hi there! I'm your AI career assistant. How can I help you today? You can ask me about job searches, resume tips, interview preparation, or companies that are known for supporting women in the Indian workplace."
    }
  ]);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [, setCurrentQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  // Add the missing variable definitions here:
  const navigationItems = [
    { id: 'home', icon: Home, label: 'Home' },
    { id: 'profile', icon: User, label: 'Profile' },
    { id: 'jobs', icon: Briefcase, label: 'Jobs' },
    { id: 'resources', icon: BookOpen, label: 'Resources' }
  ];

  const successStories = [
    {
      name: "Priya S.",
      role: "Software Engineer",
      avatar: "PS",
      quote: "Career Compass helped me negotiate a 25% higher salary than what was initially offered. The interview prep was invaluable!",
      color: "bg-purple-500"
    },
    {
      name: "Anita K.",
      role: "Product Manager",
      avatar: "AK", 
      quote: "Found my dream job at a company with amazing work-life balance. This platform understands the unique challenges women face.",
      color: "bg-indigo-500"
    },
    {
      name: "Meera T.",
      role: "Data Scientist",
      avatar: "MT",
      quote: "The AI recommendations were spot-on. Got placed at a top tech company with excellent growth opportunities.",
      color: "bg-pink-500"
    }
  ];

  const features = [
  {
    icon: Search,
    title: "Smart Job Search",
    description: "Find positions at companies known for supporting women's career growth in India",
    color: "text-purple-600",
    onClick: () => setActiveTab('jobs'), // Navigate to jobs tab
    endpoint: '/api/jobs' // Backend endpoint
  },
  {
    icon: FileText,
    title: "Resume Builder", 
    description: "Create standout resumes that highlight your achievements effectively for Indian market",
    color: "text-purple-600",
    onClick: () => setActiveTab('resources'), // Navigate to resources tab
    endpoint: '/api/resume'
  },
  {
    icon: DollarSign,
    title: "Salary Negotiation",
    description: "Get personalized advice to negotiate better pay and benefits with Indian salary standards",
    color: "text-purple-600",
    onClick: () => {
      // You could open a chat with salary-specific suggestions
      setChatMessage("Can you help me with salary negotiation tips?");
      setActiveTab('home');
    },
    endpoint: '/api/salary'
  },
];

  const suggestedQueries = [
    "Find software engineer jobs in Bangalore",
    "How to prepare for a technical interview",
    "Companies with good maternity leave policies in India",
    "Salary negotiation tips for women in tech",
    "Remote work opportunities for Indian companies"
  ];

  // Scroll to bottom of chat when new messages are added
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const sendMessage = async () => {
  if (chatMessage.trim() && !isLoading) {
    const newMessage = { type: 'user', message: chatMessage };
    setChatHistory(prev => [...prev, newMessage]);
    setChatMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: chatMessage })
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      
      // FIX: Extract the message text from the nested response object
      const assistantMessage = data.response.message || "I'm not sure how to respond to that.";
      
      setChatHistory(prev => [...prev, { 
        type: 'assistant', 
        message: assistantMessage 
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setChatHistory(prev => [...prev, { 
        type: 'assistant', 
        message: "I'm having trouble connecting to the server. Please try again later." 
      }]);
    } finally {
      setIsLoading(false);
    }
  }
};

  const handleSuggestedQuery = (query) => {
    setCurrentQuery(query);
    setChatMessage(query);
  };

  const Header = () => (
    <header className="header">
      <div className="container">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ background: 'white', padding: '0.5rem', borderRadius: '8px' }}>
              <Lightbulb size={24} color="#667eea" />
            </div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>Career Compass India</h1>
          </div>
          
          <nav style={{ display: window.innerWidth >= 768 ? 'flex' : 'none', alignItems: 'center', gap: '1.5rem' }}>
            {navigationItems.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  background: 'none',
                  border: 'none',
                  color: 'inherit',
                  cursor: 'pointer'
                }}
              >
                <item.icon size={16} />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>

          <div style={{ display: window.innerWidth >= 768 ? 'flex' : 'none', alignItems: 'center', gap: '1rem' }}>
            <button className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'none', border: '1px solid rgba(255,255,255,0.3)', color: 'white' }}>
              <LogIn size={16} />
              <span>Login</span>
            </button>
            <button className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <UserPlus size={16} />
              <span>Register</span>
            </button>
          </div>

          <button 
            style={{ 
              display: window.innerWidth < 768 ? 'block' : 'none',
              background: 'none',
              border: 'none',
              color: 'white',
              padding: '0.5rem',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile menu */}
        {isMenuOpen && (
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.3)', paddingTop: '1rem', paddingBottom: '1rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {navigationItems.map(item => (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setIsMenuOpen(false);
                  }}
                  className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    background: 'none',
                    border: 'none',
                    color: 'inherit',
                    cursor: 'pointer',
                    textAlign: 'left',
                    width: '100%'
                  }}
                >
                  <item.icon size={16} />
                  <span>{item.label}</span>
                </button>
              ))}
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.3)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                <button className="nav-item" style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%' }}>
                  <LogIn size={16} />
                  <span>Login</span>
                </button>
                <button className="nav-item" style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', marginTop: '0.25rem' }}>
                  <UserPlus size={16} />
                  <span>Register</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );

  const HeroSection = () => (
    <div className="hero">
      <div className="container">
        <div style={{ maxWidth: '900px', margin: '0 auto', textAlign: 'center' }}>
          <h1>Your AI Career Assistant</h1>
          <p>
            Welcome to Career Compass! I'm your personal AI assistant specializing in helping women find their 
            ideal jobs across Indian tech hubs. I can help with job searches, resume building, interview preparation, 
            and addressing gender-specific challenges in the Indian workplace.
          </p>
          <button className="btn-primary" style={{ fontSize: '1.125rem', padding: '1rem 2rem' }}>
            🚀 Get Started
          </button>
        </div>
      </div>
    </div>
  );

  const ChatInterface = () => (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <MessageCircle size={24} />
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Chat with Career Assistant</h3>
        </div>
        <button 
          style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.7)', cursor: 'pointer', padding: '0.25rem', borderRadius: '4px' }}
          onClick={() => {
            setChatHistory([{
              type: 'assistant',
              message: "Hi there! I'm your AI career assistant. How can I help you today? You can ask me about job searches, resume tips, interview preparation, or companies that are known for supporting women in the Indian workplace."
            }]);
          }}
        >
          <X size={20} />
        </button>
      </div>
      
      <div className="chat-container" ref={chatContainerRef}>
        {chatHistory.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.type}`}>
            <div className={`message-bubble ${msg.type}`}>
              {msg.type === 'assistant' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <div className="assistant-avatar">CC</div>
                  <span style={{ fontSize: '0.875rem', fontWeight: '500' }}>Career Assistant</span>
                </div>
              )}
              <p style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>{msg.message}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="chat-message assistant">
            <div className="message-bubble assistant">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <div className="assistant-avatar">CC</div>
                <span style={{ fontSize: '0.875rem', fontWeight: '500' }}>Career Assistant</span>
              </div>
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{ borderTop: '1px solid #e2e8f0', background: '#f8f9fa', padding: '1rem' }}>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            value={chatMessage}
            onChange={(e) => setChatMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type your message here..."
            disabled={isLoading}
            style={{
              flex: '1',
              padding: '0.75rem 1rem',
              border: '1px solid #d1d5db',
              borderRadius: '8px',
              fontSize: '0.875rem',
              outline: 'none',
              opacity: isLoading ? 0.7 : 1
            }}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !chatMessage.trim()}
            className="btn-primary"
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              padding: '0.75rem 1.5rem',
              opacity: (isLoading || !chatMessage.trim()) ? 0.7 : 1
            }}
          >
            <Send size={16} />
            <span>Send</span>
          </button>
        </div>
      </div>
    </div>
  );

  const SuggestedQueries = () => (
    <div className="card">
      <div className="card-body">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Lightbulb size={24} color="#667eea" />
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1f2937' }}>Suggested Queries</h3>
        </div>
        <div>
          {suggestedQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => handleSuggestedQuery(query)}
              className="query-button"
            >
              <Search size={16} className="query-icon" />
              <span className="query-text">{query}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const SuccessStories = () => (
    <div className="card">
      <div className="card-body">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
          <Star size={24} color="#667eea" />
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1f2937' }}>Success Stories</h3>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {successStories.map((story, index) => (
            <div key={index} className="success-story">
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                <div className={`story-avatar ${story.color}`} style={{ background: story.color === 'bg-purple-500' ? '#8b5cf6' : story.color === 'bg-indigo-500' ? '#6366f1' : '#ec4899' }}>
                  {story.avatar}
                </div>
                <div style={{ flex: '1' }}>
                  <blockquote>"{story.quote}"</blockquote>
                  <div>
                    <p className="story-author">{story.name}</p>
                    <p className="story-role">{story.role}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const FeaturesSection = () => (
  <div style={{ padding: '4rem 0', background: '#f9fafb' }}>
    <div className="container">
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '2.25rem', fontWeight: 'bold', color: '#1f2937', marginBottom: '1rem' }}>
          How Career Compass India Helps You Succeed
        </h2>
        <p style={{ fontSize: '1.25rem', color: '#6b7280' }}>
          Tools and resources designed specifically for women in the Indian workplace
        </p>
      </div>
      
      <div className="feature-grid">
        {features.map((feature, index) => (
          <div 
            key={index} 
            className="feature-card"
            onClick={feature.onClick}
            style={{ 
              cursor: 'pointer',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              // Add hover effects
              ':hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)'
              }
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px)';
              e.currentTarget.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
            }}
          >
            <div className="feature-icon">
              <feature.icon size={32} color="#667eea" />
            </div>
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
            <div style={{ 
              marginTop: '1rem', 
              color: '#667eea', 
              fontSize: '0.875rem',
              fontWeight: '500'
            }}>
              Click to explore →
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

// Add these functions to your component
const fetchJobRecommendations = async (filters = {}) => {
  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(filters)
    });
    return await response.json();
  } catch (error) {
    console.error('Error fetching jobs:', error);
    return null;
  }
};

const analyzeResume = async (resumeData) => {
  try {
    const response = await fetch('/api/resume/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(resumeData)
    });
    return await response.json();
  } catch (error) {
    console.error('Error analyzing resume:', error);
    return null;
  }
};

const getSalaryInsights = async (position, experience, location) => {
  try {
    const response = await fetch('/api/salary', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ position, experience, location })
    });
    return await response.json();
  } catch (error) {
    console.error('Error fetching salary insights:', error);
    return null;
  }
};

  const MainContent = () => {
      // Add state for job search
  const [jobFilters, setJobFilters] = useState({
    query: '',
    location: 'Bangalore',
    skills: [],
    experience: 0
  });
  const [jobResults, setJobResults] = useState([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [searchPerformed, setSearchPerformed] = useState(false);

  // Function to fetch jobs from backend
  const searchJobs = async (filters = {}) => {
    setIsLoadingJobs(true);
    setSearchPerformed(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/jobs/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filters)
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch jobs');
      }
      
      const data = await response.json();
      setJobResults(data.jobs || []);
    } catch (error) {
      console.error('Error fetching jobs:', error);
      setJobResults([]);
      
      // Fallback to sample data if API fails
      setJobResults([
        {
          '_id': '1',
          'title': 'Senior Software Engineer - Python',
          'company': 'Flipkart',
          'location': 'Bangalore',
          'experience': '3-5 years',
          'salary_range': '15-25 LPA',
          'skills_required': ['Python', 'Django', 'REST APIs', 'PostgreSQL'],
          'job_url': 'https://www.flipkartcareers.com/job123',
          'posted_date': '2024-01-15',
          'job_type': 'Full-time',
          'remote_option': false,
          'women_friendly_score': 4.5
        },
        {
          '_id': '2',
          'title': 'Frontend Developer - React',
          'company': 'Swiggy',
          'location': 'Bangalore',
          'experience': '2-4 years',
          'salary_range': '12-20 LPA',
          'skills_required': ['React', 'JavaScript', 'HTML/CSS', 'Redux'],
          'job_url': 'https://careers.swiggy.com/job456',
          'posted_date': '2024-01-14',
          'job_type': 'Full-time',
          'remote_option': true,
          'women_friendly_score': 4.3
        },
        {
          '_id': '3',
          'title': 'Data Scientist',
          'company': 'Zomato',
          'location': 'Delhi',
          'experience': '2-5 years',
          'salary_range': '18-28 LPA',
          'skills_required': ['Python', 'Machine Learning', 'SQL', 'Tableau'],
          'job_url': 'https://www.zomato.com/careers/job789',
          'posted_date': '2024-01-13',
          'job_type': 'Full-time',
          'remote_option': true,
          'women_friendly_score': 4.2
        }
      ]);
    } finally {
      setIsLoadingJobs(false);
    }
  };

  const handleSearch = () => {
    searchJobs(jobFilters);
  };

  const handleQuickSearch = (query, location = 'Bangalore') => {
    setJobFilters({
      ...jobFilters,
      query,
      location
    });
    searchJobs({ query, location });
  };
    if (activeTab === 'home') {
      return (
        <div style={{ minHeight: '100vh', background: '#f9fafb' }}>
          <HeroSection />
          
          <div className="container" style={{ padding: '4rem 0' }}>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: window.innerWidth >= 1024 ? '2fr 1fr' : '1fr', 
              gap: '2rem' 
            }}>
              <div>
                <ChatInterface />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <SuggestedQueries />
                <SuccessStories />
              </div>
            </div>
          </div>
          
          <FeaturesSection />
        </div>
      );
    }
    else if (activeTab === 'jobs') {
    return (
      <div style={{ minHeight: '100vh', background: '#f9fafb', padding: '4rem 0' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 'bold', color: '#1f2937', marginBottom: '1rem' }}>
              Job Search
            </h2>
            <p style={{ color: '#6b7280' }}>Find your next career opportunity at companies that support women's growth</p>
          </div>
          
          {/* Quick Search Buttons */}
          <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
            <h3 style={{ marginBottom: '1rem', color: '#4b5563' }}>Quick Search</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center' }}>
              {['Python Developer', 'React Developer', 'Data Scientist', 'UX Designer', 'Product Manager'].map((role) => (
                <button
                  key={role}
                  onClick={() => handleQuickSearch(role)}
                  style={{
                    padding: '0.5rem 1rem',
                    border: '1px solid #667eea',
                    borderRadius: '20px',
                    background: 'white',
                    color: '#667eea',
                    cursor: 'pointer',
                    fontSize: '0.875rem'
                  }}
                >
                  {role}
                </button>
              ))}
            </div>
          </div>

          {/* Job Search Filters */}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <div className="card-body">
              <h3 style={{ marginBottom: '1rem' }}>Search Filters</h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: window.innerWidth >= 768 ? 'repeat(2, 1fr)' : '1fr', 
                gap: '1rem',
                marginBottom: '1rem'
              }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Job Title</label>
                  <input
                    type="text"
                    placeholder="e.g., Software Engineer"
                    value={jobFilters.query}
                    onChange={(e) => setJobFilters({...jobFilters, query: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      fontSize: '0.875rem'
                    }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Location</label>
                  <input
                    type="text"
                    placeholder="e.g., Bangalore"
                    value={jobFilters.location}
                    onChange={(e) => setJobFilters({...jobFilters, location: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      fontSize: '0.875rem'
                    }}
                  />
                </div>
              </div>
              <button
                onClick={handleSearch}
                disabled={isLoadingJobs}
                className="btn-primary"
                style={{ width: '100%' }}
              >
                {isLoadingJobs ? 'Searching...' : 'Search Jobs'}
              </button>
            </div>
          </div>

          {/* Job Results */}
          {searchPerformed && (
            <div>
              <h3 style={{ marginBottom: '1rem', color: '#4b5563' }}>
                {isLoadingJobs ? 'Loading jobs...' : `Found ${jobResults.length} jobs`}
              </h3>
              
              {isLoadingJobs ? (
                <div style={{ textAlign: 'center', padding: '2rem' }}>
                  <div className="typing-indicator" style={{ justifyContent: 'center' }}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p>Searching for the best opportunities...</p>
                </div>
              ) : jobResults.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {jobResults.map((job) => (
                    <div key={job._id || job.job_id} className="card">
                      <div className="card-body">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                          <div style={{ flex: 1 }}>
                            <h4 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem', color: '#1f2937' }}>
                              {job.job_title || job.title}
                            </h4>
                            <p style={{ color: '#6b7280', marginBottom: '0.5rem' }}>
                              <strong>{job.company}</strong> • {job.location}
                            </p>
                            {job.salary_range && (
                              <p style={{ color: '#059669', fontWeight: '500', marginBottom: '0.5rem' }}>
                                💰 {job.salary_range}
                              </p>
                            )}
                            {job.experience && (
                              <p style={{ color: '#6b7280', marginBottom: '0.5rem' }}>
                                📅 {job.experience} experience
                              </p>
                            )}
                          </div>
                          {job.women_friendly_score && (
                            <div style={{ 
                              background: '#667eea', 
                              color: 'white', 
                              padding: '0.5rem', 
                              borderRadius: '8px',
                              textAlign: 'center',
                              minWidth: '60px'
                            }}>
                              <div style={{ fontSize: '0.875rem', fontWeight: 'bold' }}>{job.women_friendly_score}</div>
                              <div style={{ fontSize: '0.75rem' }}>Rating</div>
                            </div>
                          )}
                        </div>
                        
                        {job.skills_required && job.skills_required.length > 0 && (
                          <div style={{ marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                              {job.skills_required.map((skill, index) => (
                                <span
                                  key={index}
                                  style={{
                                    background: '#e5e7eb',
                                    color: '#4b5563',
                                    padding: '0.25rem 0.5rem',
                                    borderRadius: '12px',
                                    fontSize: '0.75rem'
                                  }}
                                >
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ color: '#6b7280', fontSize: '0.875rem' }}>
                            Posted {job.posted_date || 'recently'}
                          </span>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button
                              className="btn-secondary"
                              style={{ fontSize: '0.875rem', padding: '0.5rem 1rem' }}
                              onClick={() => {
                                // Save job functionality
                                console.log('Save job:', job);
                              }}
                            >
                              💾 Save
                            </button>
                            <button
                              className="btn-primary"
                              style={{ fontSize: '0.875rem', padding: '0.5rem 1rem' }}
                              onClick={() => window.open(job.job_url, '_blank')}
                            >
                              Apply Now
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card">
                  <div className="card-body" style={{ textAlign: 'center', padding: '2rem' }}>
                    <p style={{ color: '#6b7280', marginBottom: '1rem' }}>No jobs found matching your criteria.</p>
                    <button
                      className="btn-primary"
                      onClick={() => {
                        setChatMessage("Can you help me find job opportunities?");
                        setActiveTab('home');
                      }}
                    >
                      Ask the AI Assistant for Help
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

    return (
      <div style={{ minHeight: '100vh', background: '#f9fafb', padding: '4rem 0' }}>
        <div className="container">
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 'bold', color: '#1f2937', marginBottom: '1rem' }}>
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Page
            </h2>
            <p style={{ color: '#6b7280' }}>This section is under development</p>
          </div>
        </div>
      </div>
    );
  };

  const Footer = () => (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <div style={{ background: '#667eea', padding: '0.5rem', borderRadius: '8px' }}>
                <Lightbulb size={24} color="white" />
              </div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>Career Compass</h3>
            </div>
            <p style={{ color: '#a0aec0', marginBottom: '1rem' }}>
              Empowering women to navigate their professional journey with confidence and success across India.
            </p>
          </div>
          
          <div className="footer-section">
            <h4>Navigation</h4>
            <ul>
              <li><a href="#">Home</a></li>
              <li><a href="#">Jobs</a></li>
              <li><a href="#">Resources</a></li>
              <li><a href="#">About Us</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="#">Resume Templates</a></li>
              <li><a href="#">Interview Tips</a></li>
              <li><a href="#">Salary Guide</a></li>
              <li><a href="#">Career Development</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Contact Us</h4>
            <div style={{ color: '#a0aec0' }}>
              <p style={{ marginBottom: '0.5rem' }}>📧 support@careercompass.in</p>
              <p style={{ marginBottom: '0.5rem' }}>📱 +91 80 1234 5678</p>
              <p>📍 Maharashtra, India</p>
            </div>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; 2025 Career Compass. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );

  // Add CSS for typing indicator
  const styles = `
    .typing-indicator {
      display: flex;
      align-items: center;
      height: 20px;
    }
    
    .typing-indicator span {
      height: 8px;
      width: 8px;
      background-color: #667eea;
      border-radius: 50%;
      display: inline-block;
      margin: 0 2px;
      opacity: 0.6;
      animation: bounce 1.3s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(1) {
      animation-delay: 0s;
    }
    
    .typing-indicator span:nth-child(2) {
      animation-delay: 0.15s;
    }
    
    .typing-indicator span:nth-child(3) {
      animation-delay: 0.3s;
    }
    
    @keyframes bounce {
      0%, 80%, 100% {
        transform: translateY(0);
      }
      40% {
        transform: translateY(-8px);
      }
    }
  `;

  return (
    <>
      <style>{styles}</style>
      <div style={{ minHeight: '100vh', background: '#f9fafb' }}>
        <Header />
        <MainContent />
        <Footer />
      </div>
    </>
  );
};

export default CareerCompass;