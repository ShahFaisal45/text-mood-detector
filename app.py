import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Mood Detector", page_icon="🎭")

st.title("Mood Detector 🎭")
st.write("Write any sentence and the app will detect the mood.")

# Load model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Input
text = st.text_area("Enter your text:")

# Button
if st.button("Detect Mood"):
    
    if text.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        results = classifier(text)

        # Get mood with highest confidence
        best = max(results[0], key=lambda x: x["score"])

        st.subheader("Detected Mood")
        st.write(f"🎭 Mood: **{best['label']}**")
        st.write(f"📊 Confidence: **{best['score']:.2%}**")