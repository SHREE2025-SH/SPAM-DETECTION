import string
import joblib
import streamlit as st

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply the same preprocessing used during training."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


@st.cache_resource
def load_models():
    nb  = joblib.load('models/naive_bayes_spam.pkl')
    dt  = joblib.load('models/decision_tree_spam.pkl')
    vec = joblib.load('models/tfidf_vectorizer.pkl')
    return nb, dt, vec


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Detector")
st.markdown("Enter a message below and both classifiers will tell you whether it's **spam** or **ham**.")

# ── Load models ───────────────────────────────────────────────────────────────

nb_model, dt_model, vectorizer = load_models()

# ── Input ─────────────────────────────────────────────────────────────────────

message = st.text_area("Message", placeholder="Type or paste your message here…", height=150)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        cleaned    = clean_text(message)
        vectorized = vectorizer.transform([cleaned])

        nb_pred = nb_model.predict(vectorized)[0]
        dt_pred = dt_model.predict(vectorized)[0]

        nb_proba = nb_model.predict_proba(vectorized)[0]   # [ham_prob, spam_prob]

        st.markdown("---")
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Naive Bayes**")
            if nb_pred == 1:
                st.error("🚨 SPAM")
            else:
                st.success("✅ HAM (not spam)")
            st.caption(f"Confidence — Ham: {nb_proba[0]:.1%} · Spam: {nb_proba[1]:.1%}")

        with col2:
            st.markdown("**Decision Tree**")
            if dt_pred == 1:
                st.error("🚨 SPAM")
            else:
                st.success("✅ HAM (not spam)")

        st.markdown("---")
        with st.expander("Preprocessed message"):
            st.code(cleaned, language=None)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown(
        "Models trained on the **SMS Spam Collection** dataset using TF-IDF (3,000 features).\n\n"
        "- **Naive Bayes** — Multinomial NB\n"
        "- **Decision Tree** — max depth 10\n\n"
        "Run `python spam_detection.py` to retrain."
    )
    st.markdown("---")
    st.markdown("**Try an example:**")
    examples = {
        "Spam 1": "Congratulations! You've won a FREE $1,000 gift card. Click HERE to claim NOW!!!",
        "Spam 2": "URGENT: Your account has been compromised. Call us immediately to claim your prize.",
        "Ham 1":  "Hey, are we still meeting at 3pm tomorrow? Let me know!",
        "Ham 2":  "Can you pick up some milk on your way home?",
    }
    for label, text in examples.items():
        if st.button(label):
            st.session_state["_example"] = text
            st.rerun()

# Fill textarea with selected example (via session state)
if "_example" in st.session_state:
    st.text_area("Message", value=st.session_state.pop("_example"), height=150, key="example_area")
