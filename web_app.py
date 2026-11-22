import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon=":brain:")

st.title("AI Sentiment Analyzer")
st.write("Discover the sentiment behind you text!")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model=("distilbert/distilbert-base-uncased-finetuned-sst-2-english"), device=-1)
classifier = load_model()
text = st.text_area("Enter your text",placeholder="I love learning AI!")

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(text)[0] # type: ignore
        sentiment = result['label']
        confidence = result['score']

        if sentiment == "POSITIVE":
            st.success("Positive Sentiment Detected!")
        else:
            st.error("Negative Sentiment Detected!")

        st.write(f"Confidence: {confidence *100:.2f}%")
        st.progress(confidence)

        st.caption("Confidence is not a guranteed accuracy.")