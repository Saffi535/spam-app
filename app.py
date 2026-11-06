import streamlit as st
import pickle
import streamlit.components.v1 as components
import numpy as np

# Page configuration
st.set_page_config(
    page_title="CyberShield - SMS Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# Background Wallpaper & Dark Cyber Theme CSS
background_image_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80"

custom_css = f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(10, 15, 29, 0.88), rgba(10, 15, 29, 0.94)), url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Typography */
h1, h2, h3, p, span, label {{
    color: #f8fafc !important;
}}

/* Glassmorphism Metric Cards */
div[data-testid="stMetricValue"] {{
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}}

/* Input Box */
.stTextArea textarea {{
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #38bdf8 !important;
    font-size: 16px !important;
    border: 1.5px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 12px !important;
}}

/* Alert Card: Crisp contrast */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    animation: popIn 0.4s ease-out;
}}

@keyframes popIn {{
    0% {{ transform: scale(0.95); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {{
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}}

/* Button styling */
div[data-testid="stHorizontalBlock"] button {{
    background-color: rgba(30, 41, 59, 0.9) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease;
}}

div[data-testid="stHorizontalBlock"] button:hover {{
    transform: translateY(-2px);
    border-color: #38bdf8 !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Audio Speech Engine
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

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Session State for History Tracking
if 'history' not in st.session_state:
    st.session_state.history = []

try:
    model, vectorizer = load_artifacts()

    # Sidebar: Analytics & Controls
    with st.sidebar:
        st.title("🛡️ Engine Specs")
        st.markdown("**Algorithm:** Random Forest")
        st.markdown("**Vectorization:** TF-IDF Sublinear")
        st.markdown("**Environment:** Production Ready")
        st.divider()
        st.subheader("📊 Session Log")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-5:]):
                icon = "🚨" if item['pred'] == "SPAM" else "✅"
                st.write(f"{icon} **{item['pred']}** ({item['conf']:.1f}%)")
                st.caption(f'"{item["text"][:35]}..."')
        else:
            st.info("Abhi koi message test nahi kiya gaya.")

    # Main Dashboard
    st.title("🛡️ CyberShield AI: Text Classification")
    st.caption("AI-powered real-time security model to classify spam, fraud, and legitimate communications.")

    # Quick Samples
    st.markdown("##### ⚡ Quick Test Presets")
    c1, c2 = st.columns(2)
    example_text = ""
    if c1.button("📋 Sample Clean / Ham Message"):
        example_text = "Hey, are you free right now? Let's discuss the project presentation before class."
    if c2.button("🚨 Sample Malicious / Spam Offer"):
        example_text = "URGENT! You have won a 1000 cash prize or holiday voucher. Call 09061701461 to claim now. Reply STOP to cancel."

    # Input Box
    user_input = st.text_area(
        "Enter Text / SMS Message:",
        value=example_text,
        height=120,
        placeholder="Type or paste any SMS or message here..."
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("Pehle koi text message enter karein!")
        else:
            with st.spinner("Tokenizing & analyzing linguistic patterns..."):
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)[0]
                probabilities = model.predict_proba(transformed_input)[0]

            # Metrics row
            words = len(user_input.split())
            chars = len(user_input)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Words", words)
            m2.metric("Total Characters", chars)

            if prediction == 1:
                conf = probabilities[1] * 100
                m3.metric("Threat Status", "HIGH RISK", delta="SPAM", delta_color="inverse")
                st.error(f"🚨 SPAM DETECTED — Confidence: {conf:.2f}%")
                st.progress(probabilities[1])
                st.snow()
                play_voice("Warning! This message is classified as dangerous spam.")
                st.session_state.history.append({"text": user_input, "pred": "SPAM", "conf": conf})
            else:
                conf = probabilities[0] * 100
                m3.metric("Threat Status", "SAFE", delta="LEGITIMATE")
                st.success(f"✅ HAM (SAFE MESSAGE) — Confidence: {conf:.2f}%")
                st.progress(probabilities[0])
                st.balloons()
                play_voice("This message is verified as safe.")
                st.session_state.history.append({"text": user_input, "pred": "HAM", "conf": conf})

            # Top Key Features Extraction
            feature_names = np.array(vectorizer.get_feature_names_out())
            nonzero_indices = transformed_input.nonzero()[1]
            if len(nonzero_indices) > 0:
                extracted_words = feature_names[nonzero_indices][:8]
                st.markdown("##### 🔑 Key Extracted Tokens:")
                st.write(" ".join([f"`{w}`" for w in extracted_words]))

except FileNotFoundError:
    st.error("⚠️ Model artifacts missing! Ensure `rf_model.pkl` and `tfidf_vectorizer.pkl` exist.")
