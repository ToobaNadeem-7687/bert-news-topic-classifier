import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
import os
from pathlib import Path

st.set_page_config(page_title="BERT News Classifier", page_icon="📰", layout="wide")

labels = ["World", "Sports", "Business", "Sci/Tech"]

# Load model with caching
@st.cache_resource
def load_model():
    # Get absolute path
    current_dir = Path(__file__).parent
    model_path = current_dir / "news_classifier_model"
    
    if not model_path.exists():
        return None, None, f"❌ Model directory not found at: {model_path}"
    
    try:
        # Use AutoTokenizer instead of BertTokenizer for better compatibility
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = BertForSequenceClassification.from_pretrained(str(model_path))
        return model, tokenizer, None
    except Exception as e:
        return None, None, f"❌ Error loading model: {str(e)}"

# Load model
model, tokenizer, error_msg = load_model()

# Display error if model loading failed
if error_msg:
    st.error(error_msg)
else:
    st.success("✅ Model loaded successfully!")

# Sidebar for info
with st.sidebar:
    st.header("About")
    st.write("This app uses a BERT model to classify news headlines into categories: World, Sports, Business, Sci/Tech.")
    st.write("Enter a news text below and click Predict.")

# Main content
st.title("📰 BERT News Classifier")
st.markdown("### Classify News Headlines with BERT")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter News Text", height=100, placeholder="Paste or type a news headline here...")
with col2:
    st.write("")  # Spacer
    predict_button = st.button("Predict Category", type="primary")

# Prediction logic
if predict_button and text.strip():
    if model is None or tokenizer is None:
        st.error("Model not loaded. Please check the model directory.")
    else:
        with st.spinner("Classifying..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()
            category = labels[prediction]
        
        # Display result
        st.success(f"Predicted Category: **{category}**")
        
        # Confidence scores
        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
        st.write("**Confidence Scores:**")
        for i, label in enumerate(labels):
            st.progress(float(probs[i]), text=f"{label}: {float(probs[i]):.2%}")

elif predict_button and not text.strip():
    st.warning("Please enter some text to classify.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Hugging Face Transformers.*")