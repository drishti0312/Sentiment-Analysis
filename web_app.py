import streamlit as st
from transformers import pipeline

# Page configuration

st.set_page_config(page_title="Sentiment AI Analyzer", page_icon=":brain:",layout="centered", initial_sidebar_state="auto")

# Custom Styling

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f8f7ff 0%, #eef2ff 50%, #fdf4ff 100%);
    color: #1e1b4b;
}

/* Main container */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Header / Hero */
.hero {
    background: linear-gradient(135deg, #6d5dfc 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 40px 30px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(109, 93, 252, 0.22);
}

.hero h1 {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
    letter-spacing: -1px;
}

.hero p {
    color: #f3e8ff !important;
    font-size: 18px !important;
    margin-bottom: 0 !important;
}

/* Main title */
h1, h2, h3 {
    color: #312e81 !important;
}

/* Subtitle */
.stApp p {
    color: #475569;
}

/* Section heading */
[data-testid="stSubheader"] {
    color: #4338ca;
    font-weight: 700;
}

/* Input labels */
.stTextArea label,
.stSelectbox label {
    color: #3730a3 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e1b4b !important;
    -webkit-text-fill-color: #1e1b4b !important;
    caret-color: #6366f1 !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 16px !important;
    padding: 16px !important;
    font-size: 16px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.06);
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border: 1px solid #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
}

.stTextArea textarea::placeholder {
    color: #a5b4fc !important;
    -webkit-text-fill-color: #a5b4fc !important;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 12px !important;
    color: #312e81 !important;
    box-shadow: 0 3px 12px rgba(99, 102, 241, 0.05);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 20px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.22);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(99, 102, 241, 0.3);
    border: none !important;
}

/* Result card */
.result-card {
    padding: 24px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #e0e7ff;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.08);
}

.result-card h2 {
    color: #312e81 !important;
    font-size: 24px;
}

.result-card p {
    color: #64748b !important;
    font-size: 16px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e0e7ff;
    box-shadow: 0 5px 15px rgba(99, 102, 241, 0.05);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7);
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Divider */
hr {
    border-color: #ddd6fe !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 13px;
    padding-top: 25px;
}

/* Remove excessive top spacing */
[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

</style>
""", unsafe_allow_html=True)
if "history" not in st.session_state:
    st.session_state["history"] = []
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
        prediction = {"text": text, "sentiment": sentiment, "confidence": confidence}
        st.session_state["history"].append(prediction)

st.subheader("Prediction History")
st.dataframe(st.session_state["history"], use_container_width=True)
# model info

st.divider()

st.markdown("""<div class="footer">
Build with Python,Hugging Face Transformers, and Streamlit. 
<br>
Model used: distilbert-base-uncased-finetuned-sst-2-english.
</div>""", unsafe_allow_html=True)