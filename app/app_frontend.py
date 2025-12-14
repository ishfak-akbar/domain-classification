import streamlit as st
import joblib
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

st.set_page_config(page_title="News Domain Classifier", layout="centered")

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

#======UI======
st.title("Domain Classifier")
st.markdown(f"**Model Accuracy on Test Set: {MODEL_ACCURACY:.2f}%**")
st.markdown("---")

st.write("Enter any news article, headline, or paragraph to predict its domain.")

user_input = st.text_area("Input Text:", height=150, placeholder="Paste your text here...")

if st.button("Predict Domain", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Predicting..."):
            cleaned = clean_text(user_input)
            if cleaned == "":
                st.warning("Text became empty after cleaning. Try adding more content.")
            else:
                tfidf_input = vectorizer.transform([cleaned])
                prediction = model.predict(tfidf_input)[0]
                probabilities = model.predict_proba(tfidf_input)[0]
                
                prob_df = pd.DataFrame({
                    'Domain': model.classes_,
                    'Probability (%)': [f"{p*100:.2f}%" for p in probabilities]
                })
                prob_df = prob_df.sort_values('Probability (%)', ascending=False).reset_index(drop=True)

                top_prob = probabilities.max() * 100
                st.success(f"**Predicted Domain: {prediction.upper()}** ({top_prob:.2f}% confidence)")

                st.markdown("### All Domain Probabilities:")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)


st.markdown("---")
# st.caption("")