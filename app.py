import streamlit as st
import pickle

# Page styling & config
st.set_page_config(
    page_title="Text Classification - Random Forest",
    page_icon="📩",
    layout="centered"
)

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
            # Transform text through TF-IDF
            transformed_input = vectorizer.transform([user_input])

            # Model prediction
            prediction = model.predict(transformed_input)[0]
            probability = model.predict_proba(transformed_input)[0]

            st.divider()
            if prediction == 1:
                confidence = probability[1] * 100
                st.error(f"🚨 **SPAM Message Detected!**")
                st.write(f"**Confidence:** `{confidence:.2f}%`")
                st.progress(probability[1])
            else:
                confidence = probability[0] * 100
                st.success(f"✅ **HAM (Legitimate Message)**")
                st.write(f"**Confidence:** `{confidence:.2f}%`")
                st.progress(probability[0])

except FileNotFoundError:
    st.error("⚠️ Model files nahi milin! Check karein ke `rf_model.pkl` aur `tfidf_vectorizer.pkl` usi folder mein maujood hain jahan yeh `app.py` run ho rahi hai.")