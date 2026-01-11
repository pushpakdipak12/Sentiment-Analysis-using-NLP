import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="Movie Sentiment Analyzer", layout="wide")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Analyze the sentiment of movie reviews using Machine Learning!")

@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv('IMDB Dataset.csv')

model = load_model()
df = load_dataset()

with st.sidebar:
    st.header("📊 Dataset Info")
    st.write(f"Total reviews: {len(df)}")
    st.write(f"Positive: {(df['sentiment'] == 'positive').sum()}")
    st.write(f"Negative: {(df['sentiment'] == 'negative').sum()}")


col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Analyze a Review")
    user_review = st.text_area("Enter a movie review:", height=200)
    
    if st.button("🔍 Analyze"):
        if user_review.strip():
            prediction = model.predict([user_review])[0]
            confidence = model.predict_proba([user_review])[0]
            
            sentiment = "Positive ✅" if prediction == 1 else "Negative ❌"
            confidence_score = max(confidence)
            
            st.success(f"**Sentiment: {sentiment}**")
            st.info(f"**Confidence: {confidence_score:.2%}**")
        else:
            st.warning("Please enter a review!")

with col2:
    st.subheader("📈 Dataset Statistics")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#28a745', '#dc3545']
    ax.bar(['Positive', 'Negative'], [sentiment_counts.get('positive', 0), sentiment_counts.get('negative', 0)], color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

st.subheader("📚 Sample Reviews from Dataset")
col_pos, col_neg = st.columns(2)

with col_pos:
    st.write("**Positive Review Example:**")
    positive_sample = df[df['sentiment'] == 'positive']['review'].iloc[0]
    st.info(positive_sample[:300] + "...")

with col_neg:
    st.write("**Negative Review Example:**")
    negative_sample = df[df['sentiment'] == 'negative']['review'].iloc[0]
    st.error(negative_sample[:300] + "...")