import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned FinBERT model and tokenizer
@st.cache_resource  
def load_model():
    model_path = r"C:\Users\MartinMuru\Documents\FinetunedFinbertTwi\finbert_finetuned\finbert_finetuned"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

tokenizer, model = load_model()

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_mapping[predicted_class]

# Streamlit UI
st.title("📊 Kenyan Stock News Sentiment Analysis")

st.markdown("""
This tool analyzes the sentiment of **Kenyan stock news headlines** using a fine-tuned **FinBERT** model.  
Simply enter a **headline**, and the model will predict whether it is **positive, neutral, or negative**.
""")

user_input = st.text_area("Enter a news headline:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a valid news headline.")

st.write("*Finetuned using Kenyan stock news data*")


