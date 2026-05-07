import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Fake Job Detector for Call Center/Scam Jobs")
st.markdown("Paste a job post below to analyze its legitimacy and identify potential red flags.")

# Initialize session state for model and vectorizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Load or create a simple model
def load_model():
    """Create or load a simple ML model for demonstration"""
    # For a real application, you'd train this on actual scam/legitimate job data
    # This is a simplified version for demonstration
    
    # Sample data (in reality, you'd have a proper dataset)
    sample_scam_texts = [
        "Earn $5000 weekly working from home with no experience needed",
        "URGENT HIRING!!! WhatsApp us at +1234567890 for interview",
        "Immediate start, no interview required, just send your personal details",
        "High salary with zero qualifications needed",
        "Contact us on Telegram @easyjob for more information",
        "Get paid daily through cash app or PayPal",
        "Work only 2 hours daily and earn $3000 monthly",
        "No resume needed, just WhatsApp your name and address",
        "Guaranteed income with minimal effort",
        "EASY MONEY!!! Fast hiring process"
    ]
    
    sample_legit_texts = [
        "We are looking for qualified candidates with relevant experience",
        "Please submit your resume through our official careers portal",
        "Competitive salary based on experience and qualifications",
        "The interview process will consist of multiple rounds",
        "Successful candidates will receive benefits including health insurance",
        "Minimum 2 years of experience in customer service required",
        "Submit your application through our company website",
        "Background check will be conducted for selected candidates",
        "This position requires a bachelor's degree or equivalent",
        "We offer professional development opportunities"
    ]
    
    # Combine and label
    texts = sample_scam_texts + sample_legit_texts
    labels = [1] * len(sample_scam_texts) + [0] * len(sample_legit_texts)
    
    # Create and train vectorizer and model
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression()
    model.fit(X, labels)
    
    return model, vectorizer

# Rule-based detection functions
def detect_urgency(text):
    """Detect urgency indicators in text"""
    urgency_keywords = [
        'urgent', 'immediate', 'asap', 'quick', 'fast', 'instant',
        'hiring immediately', 'start now', 'right away', 'limited time',
        'don\'t miss out', 'apply now', 'last chance'
    ]
    
    matches = []
    for keyword in urgency_keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            matches.append(keyword)
    
    return matches

def detect_salary_language(text):
    """Detect unrealistic salary promises"""
    salary_patterns = [
        r'\$\d{3,5}\s*(weekly|daily|per week|per day)',
        r'earn\s+\$\d{3,}\s*(easily|quickly|fast)',
        r'high\s+salary\s+no\s+experience',
        r'guaranteed\s+(income|salary|payment)',
        r'\$\d{2,5}k?\s*(monthly|per month)',
        r'make\s+money\s+fast',
        r'get\s+rich\s+quick'
    ]
    
    matches = []
    for pattern in salary_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            matches.extend(found)
    
    return matches

def detect_contact_methods(text):
    """Detect suspicious contact methods"""
    contact_patterns = [
        r'whatsapp',
        r'telegram',
        r'@[\w]+',  # Generic handle
        r'\+\d{10,}',  # Phone numbers
        r'cash\s*app',
        r'paypal\s*me',
        r'direct\s*message',
        r'DM\s*us',
        r'message\s*us\s*on'
    ]
    
    matches = []
    for pattern in contact_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            matches.extend(found)
    
    return matches

def detect_grammar_issues(text):
    """Detect poor grammar and excessive punctuation"""
    issues = []
    
    # Excessive capitalization
    if re.search(r'[A-Z]{4,}', text):
        issues.append('Excessive capitalization')
    
    # Multiple exclamation marks
    if re.search(r'!{2,}', text):
        issues.append('Multiple exclamation marks')
    
    # Poor sentence structure
    lines = text.split('\n')
    short_lines = [line for line in lines if len(line.split()) < 3 and len(line) > 10]
    if len(short_lines) > 3:
        issues.append('Poor sentence structure')
    
    # Missing professional language
    professional_terms = ['experience', 'qualifications', 'resume', 'interview', 'position']
    if not any(term in text.lower() for term in professional_terms):
        issues.append('Lacks professional terminology')
    
    return issues

def detect_too_good_to_be_true(text):
    """Detect offers that seem too good to be true"""
    patterns = [
        r'no\s+experience\s+needed',
        r'work\s+from\s+home',
        r'flexible\s+hours',
        r'high\s+pay',
        r'easy\s+money',
        r'get\s+paid\s+to',
        r'no\s+interview',
        r'no\s+resume',
        r'instant\s+hiring',
        r'guaranteed\s+job'
    ]
    
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            matches.extend(found)
    
    return matches

def calculate_scam_score(red_flags):
    """Calculate scam probability based on red flags"""
    base_score = 0
    weights = {
        'urgency': 15,
        'salary': 20,
        'contact': 25,
        'grammar': 10,
        'too_good': 20,
        'ml_prediction': 10
    }
    
    for flag_type, flag_list in red_flags.items():
        if flag_type != 'score' and flag_list:
            if flag_type == 'ml_prediction':
                base_score += flag_list[0] * weights[flag_type]
            else:
                base_score += min(len(flag_list) * 5, weights[flag_type])
    
    return min(100, base_score)

# Main analysis function
def analyze_job_post(text):
    """Analyze job post and return red flags"""
    if not text or len(text.strip()) < 20:
        return None
    
    # Initialize red flags dictionary
    red_flags = {
        'urgency': detect_urgency(text),
        'salary': detect_salary_language(text),
        'contact': detect_contact_methods(text),
        'grammar': detect_grammar_issues(text),
        'too_good': detect_too_good_to_be_true(text),
        'ml_prediction': []
    }
    
    # Get ML prediction if model is loaded
    if st.session_state.model and st.session_state.vectorizer:
        try:
            transformed = st.session_state.vectorizer.transform([text])
            prediction = st.session_state.model.predict_proba(transformed)[0][1]
            red_flags['ml_prediction'] = [prediction]
        except:
            red_flags['ml_prediction'] = [0.5]  # Neutral if error
    
    # Calculate score
    red_flags['score'] = calculate_scam_score(red_flags)
    
    return red_flags

# Explanation generator
def generate_explanation(red_flags):
    """Generate explanation based on red flags"""
    explanations = []
    
    if red_flags['urgency']:
        explanations.append(f"🚨 **Urgency detected**: The post uses urgent language ({', '.join(red_flags['urgency'][:3])}). Legitimate jobs rarely pressure applicants.")
    
    if red_flags['salary']:
        explanations.append(f"💰 **Unrealistic salary**: Mentions unrealistic earnings ({', '.join(red_flags['salary'][:3])}). Scams often promise high pay for little work.")
    
    if red_flags['contact']:
        explanations.append(f"📱 **Suspicious contact methods**: Uses informal channels ({', '.join(red_flags['contact'][:3])}). Legitimate companies use professional communication.")
    
    if red_flags['grammar']:
        explanations.append(f"✍️ **Language issues**: Shows signs of poor professionalism ({', '.join(red_flags['grammar'])}). Professional job posts are well-written.")
    
    if red_flags['too_good']:
        explanations.append(f"🎯 **Too good to be true**: Promises unrealistic benefits ({', '.join(red_flags['too_good'][:3])}). Be wary of offers requiring no experience for high pay.")
    
    if red_flags.get('ml_prediction', [0])[0] > 0.7:
        explanations.append(f"🤖 **AI detection**: Our machine learning model identified patterns consistent with scam job posts.")
    
    if not explanations:
        explanations.append("✅ No major red flags detected. However, always verify the company through official channels.")
    
    return explanations

# Load the model
if st.session_state.model is None:
    with st.spinner("Loading detection model..."):
        st.session_state.model, st.session_state.vectorizer = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Analyze Job Post", "Red Flag Guide", "About"])

with tab1:
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_text = st.text_area(
            "Paste the job post here:",
            height=300,
            placeholder="Example: URGENT HIRING!!! Work from home and earn $5000 weekly. No experience needed. WhatsApp us at +1234567890 for immediate start..."
        )
    
    with col2:
        st.markdown("### What to look for:")
        st.markdown("""
        - 🚨 Urgent hiring language
        - 💰 Unrealistic salary promises
        - 📱 Contact via WhatsApp/Telegram
        - ❗ Poor grammar & excessive punctuation
        - 🎯 'Too good to be true' offers
        - 🏢 Lack of company details
        """)
        
        st.markdown("### Example scam indicators:")
        st.markdown("""
        ```
        "EARN $5000 WEEKLY!"
        "WhatsApp for interview"
        "No experience needed"
        "Immediate start"
        ```
        """)
    
    # Analyze button
    if st.button("Analyze Job Post", type="primary", use_container_width=True):
        if job_text:
            with st.spinner("Analyzing job post..."):
                # Analyze the text
                red_flags = analyze_job_post(job_text)
                
                if red_flags:
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        score = red_flags['score']
                        if score < 30:
                            st.metric("Scam Probability", f"{score}%", delta="LOW RISK", delta_color="normal")
                        elif score < 70:
                            st.metric("Scam Probability", f"{score}%", delta="MEDIUM RISK", delta_color="off")
                        else:
                            st.metric("Scam Probability", f"{score}%", delta="HIGH RISK", delta_color="inverse")
                    
                    with col2:
                        st.metric("Red Flags Found", len([v for k, v in red_flags.items() if k != 'score' and v]))
                    
                    with col3:
                        urgency_count = len(red_flags['urgency'])
                        st.metric("Urgency Indicators", urgency_count)
                    
                    with col4:
                        contact_count = len(red_flags['contact'])
                        st.metric("Suspicious Contacts", contact_count)
                    
                    # Progress bar for score
                    st.progress(red_flags['score'] / 100)
                    
                    # Detailed breakdown
                    st.subheader("📋 Detailed Breakdown")
                    
                    explanations = generate_explanation(red_flags)
                    for explanation in explanations:
                        st.info(explanation)
                    
                    # Show specific findings
                    with st.expander("View specific detected issues"):
                        for flag_type, flag_list in red_flags.items():
                            if flag_type != 'score' and flag_list and flag_type != 'ml_prediction':
                                st.write(f"**{flag_type.title()}**:")
                                for item in flag_list[:5]:  # Show first 5 items
                                    st.write(f"  - {item}")
                    
                    # Recommendations
                    st.subheader("🛡️ Recommendations")
                    
                    if red_flags['score'] > 70:
                        st.error("**HIGH RISK DETECTED** - This job post shows multiple scam indicators:")
                        st.markdown("""
                        1. **DO NOT** share personal information
                        2. **DO NOT** pay any "registration" or "training" fees
                        3. **DO NOT** contact through suspicious channels
                        4. Research the company on official platforms
                        5. Report suspicious posts to authorities
                        """)
                    elif red_flags['score'] > 30:
                        st.warning("**CAUTION ADVISED** - Verify this opportunity carefully:")
                        st.markdown("""
                        1. Research the company's official website
                        2. Check employee reviews on Glassdoor/LinkedIn
                        3. Ensure communication is through professional channels
                        4. Never pay money to get a job
                        5. Trust your instincts
                        """)
                    else:
                        st.success("**LOW RISK** - This appears legitimate, but always verify:")
                        st.markdown("""
                        1. Still research the company
                        2. Ensure proper interview process
                        3. Verify job offer through official channels
                        4. Read the contract carefully
                        """)
                else:
                    st.error("Please enter a longer job post for analysis.")
        else:
            st.warning("Please paste a job post to analyze.")

with tab2:
    st.header("Red Flag Guide for Job Seekers")
    
    st.subheader("Common Scam Tactics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Urgency & Pressure
        - "Hiring immediately"
        - "Limited time offer"
        - "Must start today"
        - "Quick hiring process"
        
        ### 💰 Unrealistic Promises
        - "Earn $5000 weekly from home"
        - "No experience needed"
        - "Guaranteed income"
        - "Get rich quick schemes"
        
        ### 📱 Unprofessional Contact
        - WhatsApp/Telegram interviews
        - Generic email addresses
        - No company phone number
        - Direct messages only
        """)
    
    with col2:
        st.markdown("""
        ### ❗ Poor Presentation
        - Bad grammar/spelling
        - Excessive punctuation!!!!
        - ALL CAPS TEXT
        - Lack of company details
        
        ### 🎯 Vague Requirements
        - No specific skills needed
        - "Anyone can do it"
        - No interview process
        - No background check
        
        ### 💸 Request for Money
        - "Training fee"
        - "Registration cost"
        - "Equipment deposit"
        - "Background check fee"
        """)
    
    st.subheader("Legitimate Job Indicators")
    st.markdown("""
    - ✅ Clear company information and website
    - ✅ Professional email addresses (name@company.com)
    - ✅ Structured interview process
    - ✅ Realistic salary ranges
    - ✅ Specific job requirements
    - ✅ Professional communication
    - ✅ No request for money
    - ✅ Physical office address (or verified remote)
    """)
    
    st.subheader("Safety Checklist")
    checklist = st.columns(2)
    
    with checklist[0]:
        st.checkbox("Company has official website")
        st.checkbox("Job posted on reputable platform")
        st.checkbox("Professional email/contact provided")
        st.checkbox("Realistic job requirements")
    
    with checklist[1]:
        st.checkbox("No money requested upfront")
        st.checkbox("Clear interview process")
        st.checkbox("Physical address verifiable")
        st.checkbox("Employee reviews available")

with tab3:
    st.header("About Fake Job Detector")
    
    st.markdown("""
    ### How It Works
    
    This tool uses a combination of techniques to identify potential scam job posts:
    
    1. **Rule-based Detection**: Identifies specific patterns and keywords commonly used in scams
    2. **Machine Learning**: Analyzes text patterns using a trained classifier
    3. **Heuristic Analysis**: Checks for multiple scam indicators
    
    ### Detection Methods
    
    - **Salary-Language Tricks**: Identifies unrealistic earnings promises
    - **Grammar & Urgency Analysis**: Detects poor writing and pressure tactics
    - **Contact Method Verification**: Flags suspicious communication channels
    - **Too-Good-To-Be-True Detection**: Identifies unrealistic offers
    
    ### Technology Stack
    
    - **Python** with Natural Language Processing (NLP)
    - **Regex Patterns** for rule-based detection
    - **Machine Learning Classifier** (Logistic Regression)
    - **Streamlit** for web interface
    
    ### Important Disclaimer
    
    This tool provides **risk assessment only** and is not 100% accurate. Always:
    
    - Conduct your own research
    - Verify through official channels
    - Never share sensitive information
    - Report suspicious posts to authorities
    
    ### For Employers
    
    If you're a legitimate employer and your posts are flagged:
    
    1. Use professional language in job descriptions
    2. Provide clear company information
    3. Use official communication channels
    4. Avoid excessive urgency language
    """)
    
    st.info("⚠️ **Note**: This is a demonstration tool. For production use, the model should be trained on a larger, real-world dataset of scam and legitimate job posts.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔍 Use this tool as a first line of defense against job scams. Always verify opportunities through official channels.</p>
        <p><small>Note: This tool is for educational purposes. Accuracy may vary.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)