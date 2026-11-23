import streamlit as st
from transformers import pipeline

# Page configuration

st.set_page_config(page_title="Sentiment AI Analyzer", page_icon=":brain:",layout="centered", initial_sidebar_state="auto")

# Custom Styling
st.markdown(""" <style>
.stApp{
background-color: #f5f7fb;
color:#17233b;
}

.block-container{
max-width: 850px;
padding-top: 2rem;
}

.hero {
background: linear-gradient(120deg, #172554, #2563eb);
padding: 35px 25px;
border-radius: 18px;
text-align: center;
margin-bottom: 30px;
}
.hero h1 {
color : white;
font-size: 38px;
margin-bottom: 10px;
}
.hero p {
color: black;
font-size: 38px;
margin-bottom: 10px;
}
.stButton>button[kind="primary"] {
background-color: #2563eb;
color: white;
border-radius: 8px;
font-weight:bold;
}

.result-card {
padding: 20px;
border-radius: 10px;
background: white;
border: 1px solid #e2e8f0;
margin-top: 20px;
}
.result-card h2{
color: #17233b;
}
.result-card p{
color: #475569;
}
.footer {
text-align: center;
color: #64748b;
font-size: 14px;
padding-top: 30px;
}
.st-key-example_select label p{
color: #172554 !important;
font-weight: 600 !important;
font-size: 16px !important;}
/* Text area styling */
.stTextArea textarea {
    background-color: white !important;
    color: #17233b !important;
    -webkit-text-fill-color: #17233b !important;
    caret-color: #17233b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}
</style> """, unsafe_allow_html=True)

# Header 
st.markdown("""
<div class="hero">
<h1>Sentiment Analyzer</h1>
<p>Understand the sentiments behind your words! </p>
</div>
""", unsafe_allow_html=True)

st.title("AI Sentiment Analyzer")
st.write("Discover the sentiment behind you text!")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model=("distilbert/distilbert-base-uncased-finetuned-sst-2-english"), device=-1)

with st.spinner("Loading model..."):
    classifier = load_model()

# Example sentences
example_sentences = {
    "custome text":"",
    "Positive":"I love learning AI!",
    "Negative":"This is the worst experience I've ever had.",
    "Positive": "I'm feeling great today!",
    "Negative": "I hate waiting in long lines.",
    "Positive": "The movie was fantastic and thrilling!"
}
st.subheader("Analyze your text!")

selected = st.selectbox("Choose an example or enter your own text", options=list(example_sentences.keys()),key="example_select")

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# update text when example is selected

if st.session_state.get("last_example") != selected:
    st.session_state.text_input = example_sentences[selected]
    st.session_state.last_example = selected
text = st.text_area("Enter your text",key="text_input",height=130,placeholder="Type your sentence here...")

# Analyse sentiments

if st.button("Analyze Sentiment",type="primary",use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(text,truncation=True)[0]
            sentiment = result['label']
            confidence = result['score']
            st.markdown(f"""
            <div class="result-card">
                <h2>Analysis Result</h2>
                <p>Prediction generated successfully!</p>
            </div>
            """, unsafe_allow_html=True)
        if sentiment == "POSITIVE":
            st.success("Positive Sentiment Detected!")
        else:
            st.error("Negative Sentiment Detected!")
        st.metric(label="Model Confidence", value=f"{confidence*100:.2}%")

        st.write(f"Confidence: {confidence *100:.2f}%")
        st.progress(confidence)

        st.caption("Confidence is not a guranteed accuracy!")

# model info

st.divider()

st.markdown("""<div class="footer">
Build with Python,Hugging Face Transformers, and Streamlit. 
<br>
Model used: distilbert-base-uncased-finetuned-sst-2-english.
</div>""", unsafe_allow_html=True)