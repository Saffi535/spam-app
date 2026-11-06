import streamlit as st
import pickle
import streamlit.components.v1 as components

# Page styling & config
st.set_page_config(
    page_title="Text Classification - Random Forest",
    page_icon="📩",
    layout="centered"
)

# 1. Custom Background Wallpaper via CSS
background_image_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80"

custom_css = f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(0, 0, 0, 0.72), rgba(0, 0, 0, 0.72)), url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Card container and text readability */
.stMarkdown, h1, h2, h3, p, span, label {{
    color: #f1f5f9 !important;
}}

.stTextArea textarea {{
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 8px;
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
