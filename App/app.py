import streamlit as st
import pickle
import nltk
import string
import os
import logging
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🛡️ Smart Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2e4057;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #e57373;
    }
    .not-spam {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #81c784;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models with error handling and caching"""
    try:
        if not os.path.exists('vectorizer.pkl'):
            st.error("❌ vectorizer.pkl file not found!")
            return None, None
        if not os.path.exists('model.pkl'):
            st.error("❌ model.pkl file not found!")
            return None, None
            
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Models loaded successfully")
        return tfidf, model
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

@st.cache_data
def download_nltk_data():
    """Download required NLTK data with caching"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("📥 Downloading required language data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        st.success("✅ Language data downloaded successfully!")

@st.cache_data
def get_stopwords():
    """Get stopwords with caching"""
    try:
        return set(stopwords.words('english'))
    except Exception as e:
        logger.error(f"Error loading stopwords: {e}")
        return set()

def validate_input(text):
    """Validate input text"""
    if not text or text.strip() == "":
        return False, "Please enter a message to analyze."
    
    if len(text.strip()) < 3:
        return False, "Message is too short. Please enter at least 3 characters."
    
    if len(text) > 5000:
        return False, "Message is too long. Please keep it under 5000 characters."
    
    return True, ""

def preprocess_text(text):
    """Enhanced text preprocessing with better error handling"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        text = nltk.word_tokenize(text)
        
        # Keep only alphanumeric tokens
        tokens = [token for token in text if token.isalnum()]
        
        # Remove stopwords and punctuation
        stop_words = get_stopwords()
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        
        # Apply stemming
        ps = PorterStemmer()
        tokens = [ps.stem(token) for token in tokens]
        
        return " ".join(tokens)
    
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return ""

def get_prediction_details(model, vector_input):
    """Get prediction with confidence score"""
    try:
        prediction = model.predict(vector_input)[0]
        
        # Try to get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vector_input)[0]
            confidence = max(probabilities) * 100
        elif hasattr(model, 'decision_function'):
            decision_score = model.decision_function(vector_input)[0]
            confidence = abs(decision_score) * 10  # Normalize roughly
            confidence = min(confidence, 100)  # Cap at 100%
        else:
            confidence = 75  # Default confidence if no probability method available
            
        return prediction, confidence
    except Exception as e:
        logger.error(f"Error getting prediction details: {e}")
        return None, 0

def display_prediction_result(prediction, confidence, message_length, processed_length):
    """Display prediction results with enhanced visualization"""
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box spam">
            🚨 SPAM DETECTED
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar for spam
        confidence_color = "#f44336" if confidence > 80 else "#ff9800" if confidence > 60 else "#ffc107"
        st.markdown(f"""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336; color: #000000;">
            <strong style="color: #000000;">⚠️ This message appears to be spam</strong><br>
            <div style="margin-top: 10px;">
                <div style="background-color: {confidence_color}; width: {confidence}%; height: 10px; border-radius: 5px;"></div>
                <small style="color: #000000;">Confidence: {confidence:.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="prediction-box not-spam">
            ✅ LEGITIMATE MESSAGE
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar for legitimate
        confidence_color = "#4caf50" if confidence > 80 else "#8bc34a" if confidence > 60 else "#cddc39"
        st.markdown(f"""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; color: #000000;">
            <strong style="color: #000000;">✅ This message appears to be legitimate</strong><br>
            <div style="margin-top: 10px;">
                <div style="background-color: {confidence_color}; width: {confidence}%; height: 10px; border-radius: 5px;"></div>
                <small style="color: #000000;">Confidence: {confidence:.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Download NLTK data
    download_nltk_data()
    
    # Load models
    tfidf, model = load_models()
    
    if tfidf is None or model is None:
        st.stop()
    
    # Main header
    st.markdown('<h1 class="main-header">🛡️ Smart Spam Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter your message")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type message", "Upload file"])
        
        input_message = ""
        
        if input_method == "Type message":
            input_message = st.text_area(
                "Message to analyze:",
                height=150,
                placeholder="Enter the email or SMS message you want to check for spam...",
                help="Enter any email or SMS message to check if it's spam or legitimate."
            )
        else:
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                input_message = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", input_message, height=150, disabled=True)
        
        # Character count
        if input_message:
            char_count = len(input_message)
            word_count = len(input_message.split())
            st.caption(f"📊 Characters: {char_count} | Words: {word_count}")
        
        # Prediction button
        predict_button = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
        
        if predict_button:
            # Validate input
            is_valid, error_msg = validate_input(input_message)
            
            if not is_valid:
                st.error(error_msg)
            else:
                with st.spinner("🤖 Analyzing message..."):
                    try:
                        # Preprocess the text
                        processed_message = preprocess_text(input_message)
                        
                        if not processed_message:
                            st.warning("⚠️ No meaningful content found after preprocessing.")
                        else:
                            # Vectorize the text
                            vector_input = tfidf.transform([processed_message]).toarray()
                            
                            # Get prediction with confidence
                            prediction, confidence = get_prediction_details(model, vector_input)
                            
                            if prediction is not None:
                                # Display results
                                display_prediction_result(
                                    prediction, confidence, 
                                    len(input_message), len(processed_message)
                                )
                                
                                # Add to history (in session state)
                                if 'prediction_history' not in st.session_state:
                                    st.session_state.prediction_history = []
                                
                                st.session_state.prediction_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'message': input_message[:100] + "..." if len(input_message) > 100 else input_message,
                                    'prediction': 'Spam' if prediction == 1 else 'Legitimate',
                                    'confidence': f"{confidence:.1f}%"
                                })
                            else:
                                st.error("❌ Error making prediction. Please try again.")
                                
                    except Exception as e:
                        st.error(f"❌ An error occurred during analysis: {str(e)}")
                        logger.error(f"Prediction error: {e}")
    
    with col2:
        # Sidebar information
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Info")
        
        if input_message:
            char_count = len(input_message)
            word_count = len(input_message.split())
            processed_text = preprocess_text(input_message)
            processed_words = len(processed_text.split()) if processed_text else 0
            
            st.metric("Original Characters", char_count)
            st.metric("Original Words", word_count)
            st.metric("Processed Words", processed_words)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips section
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("💡 Tips")
        st.markdown("""
        **Common spam indicators:**
        - Urgent language ("Act now!")
        - Suspicious links
        - Poor grammar/spelling
        - Requests for personal info
        - Too-good-to-be-true offers
        
        **Stay safe:**
        - Never click suspicious links
        - Don't share personal information
        - Verify sender identity
        - Trust your instincts
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Recent Analysis History")
        
        # Create DataFrame for history
        df = pd.DataFrame(st.session_state.prediction_history)
        df = df.iloc[::-1]  # Reverse to show latest first
        
        # Display as table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

if __name__ == "__main__":
    main()