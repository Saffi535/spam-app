import streamlit as st
import pickle
import streamlit.components.v1 as components

# Page styling & config
st.set_page_config(
    page_title="Text Classification - Random Forest",
    page_icon="📩",
    layout="centered"
)

# 1. Background Wallpaper & Modern Glass UI CSS
background_image_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80"

custom_css = f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(10, 15, 29, 0.82), rgba(10, 15, 29, 0.88)), url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Clean readable headings & text */
h1, h2, h3, p, span, label {{
    color: #f8fafc !important;
}}

/* Dark Glassmorphism Input Textarea */
.stTextArea textarea {{
    background-color: rgba(15, 23, 42, 0.75) !important;
    color: #38bdf8 !important;
    font-size: 16px !important;
    border: 1.5px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px) !important;
}}

.stTextArea textarea:focus {{
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important;
}}

/* Quick Action Secondary Buttons */
div[data-testid="stHorizontalBlock"] button {{
    background-color: rgba(30, 41, 59, 0.85) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}}

div[data-testid="stHorizontalBlock"] button:hover {{
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Text to Speech Function (Browser Voice)
def play_voice(text):
    clean_text = text.replace("'", "").replace('"', '')
    js_code = f"""
    <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{clean_text}');
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0, width=0)

# Title & Description
st.title("📩 SMS & Text Spam Classifier")
st.write("Yeh application text message ko classify karti hai using **TF-IDF Vectorization** aur **Random Forest Classifier**.")

# Model & Vectorizer load function
@st.cache_resource
def load_artifacts():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()

    # Quick test examples
    st.markdown("### Test Example Message")
    col1, col2 = st.columns(2)
    example_text = ""
    if col1.button("📋 Sample Normal Message"):
        example_text = "Hey, are you coming to campus today for the lecture?"
    if col2.button("🚨 Sample Spam Message"):
        example_text = "Congratulations! You have won a 1000 cash prize. Call this hotline right now to claim."

    # Input Area
    user_input = st.text_area(
        "Enter Text / SMS below:",
        value=example_text,
        height=130,
        placeholder="Type or paste your message here..."
    )

    # Predict Button
    if st.button("Predict Classification", type="primary"):
        if not user_input.strip():
            st.warning("Pehle koi text message enter karein!")
        else:
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]
            probability = model.predict_proba(transformed_input)[0]

            st.divider()
            if prediction == 1:
                confidence = probability[1] * 100
                st.error(f"🚨 **SPAM Message Detected!**")
                st.write(f"**Confidence:** `{confidence:.2f}%`")
                st.progress(probability[1])
                play_voice("Alert! This message is classified as Spam.")
            else:
                confidence = probability[0] * 100
                st.success(f"✅ **HAM (Legitimate Message)**")
                st.write(f"**Confidence:** `{confidence:.2f}%`")
                st.progress(probability[0])
                play_voice("This message is legitimate and safe.")

except FileNotFoundError:
    st.error("⚠️ Model files nahi milin! Check karein ke `rf_model.pkl` aur `tfidf_vectorizer.pkl` repo mein upload hain.")
