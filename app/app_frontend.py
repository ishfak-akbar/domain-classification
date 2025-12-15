import streamlit as st
import joblib
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

st.set_page_config(page_title="News Domain Classifier", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: #1c1c1c;
        color: #e0e0e0;
    }

    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #088f8f, #0ab8b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #bbbbbb;
        margin-bottom: 1rem;
    }

    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #088f8f33, #088f8f66);
        color: #ffffff;
        padding: 0.4rem 0.9rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 0.9rem;
        border: 1px solid #088f8f;
        box-shadow: 0 3px 10px rgba(8, 143, 143, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #088f8f, #0ab8b8);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 900 !important;
        font-size: 1rem !important;
        box-shadow: 0 5px 15px rgba(8, 143, 143, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0ab8b8, #088f8f);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(8, 143, 143, 0.5);
    }

    .prediction-box {
        padding: 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1c1c1c, #252525);
        border: 2px solid #088f8f;
        box-shadow: 0 6px 20px rgba(8, 143, 143, 0.25);
        text-align: center;
        margin: 1.2rem 0;
    }
    .domain-label {
        font-size: 2rem;
        font-weight: bold;
        color: #088f8f;
        text-shadow: 0 0 12px rgba(8, 143, 143, 0.5);
        margin-bottom: 0.4rem;
    }
    .confidence {
        font-size: 1.1rem;
        color: #cccccc;
    }

    .stTextArea > div > div > textarea {
        background-color: #252525;
        color: #e0e0e0;
        border: 1px solid #088f8f;
        border-radius: 10px;
        font-size: 0.95rem;
    }

    .stDataFrame {
        font-size: 0.9rem;
    }

    h1, h2, h3, h4, p, div, span, label {
        color: #e0e0e0 !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    hr {
        border-color: #088f8f44;
        margin: 0rem 0;
    }
    .block-container {
        padding-top: 4rem;   
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_ACCURACY = 85.74

@st.cache_resource
def load_model():
    model = joblib.load('domain_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

nltk.data.path.append("./nltk_data")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# ====== UI ======
st.markdown('<h1 class="main-header">News Domain Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quickly determine the domain of any news article, headline, or paragraph using intelligent text analysis.</p>', unsafe_allow_html=True)

st.markdown(f'<div style="text-align: center; margin: 1rem 0;"><span class="accuracy-badge">Model Accuracy: {MODEL_ACCURACY:.2f}%</span></div>', unsafe_allow_html=True)

st.markdown("---")

user_input = st.text_area(
    "📝 Enter your news text here:",
    height=160, 
    placeholder="Paste a news headline, paragraph, or full article..."
)

if st.button("🔮 Predict Domain", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text and predicting..."):
            cleaned = clean_text(user_input)
            if not cleaned:
                st.warning("⚠️ The text became empty after cleaning. Try adding more content.")
            else:
                tfidf_input = vectorizer.transform([cleaned])
                prediction = model.predict(tfidf_input)[0]
                probabilities = model.predict_proba(tfidf_input)[0]

                prob_df = pd.DataFrame({
                    'Domain': model.classes_,
                    'Probability (%)': [f"{p*100:.2f}%" for p in probabilities]
                }).sort_values('Probability (%)', ascending=False).reset_index(drop=True)

                top_prob = probabilities.max() * 100

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="domain-label">{prediction.upper()}</div>
                        <div class="confidence">with {top_prob:.2f}% confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 📊 All Domain Probabilities")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built using Streamlit")