import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
# ── Step 2: Load ────────────────────────────────────────────────────────────
df = pd.read_csv('spam.csv', encoding='latin-1')

# ── Step 3: Fix columns & Explore ──────────────────────────────────────────

# Keep only the useful columns (v1 = label, v2 = message)
df = df[['v1', 'v2']]

# Rename to meaningful names
df.columns = ['label', 'message']

# Check result
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# How many spam vs ham?
print("\nLabel distribution:")
print(df['label'].value_counts())

# Any missing values?
print("\nMissing values:")
print(df.isnull().sum())

# ── Step 4: Text Cleaning ───────────────────────────────────────────────────
import string

def clean_text(text):
    # 1. Lowercase everything
    text = text.lower()

    # 2. Remove punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Apply the function to every row in the message column
df['clean_message'] = df['message'].apply(clean_text)

# Compare original vs cleaned
print("\nOriginal vs Cleaned (first 3 rows):")
for i in range(3):
    print(f"\n[{df['label'][i].upper()}]")
    print("  Before:", df['message'][i])
    print("  After: ", df['clean_message'][i])

# ── Step 5: Encode Labels, Split, then TF-IDF ──────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Encode labels: spam → 1, ham → 0
y = df['label'].map({'spam': 1, 'ham': 0})
print("\nEncoded labels (first 5):", y[:5].tolist())

# Features (raw text — vectorizer will convert to numbers)
X = df['clean_message']

# Split FIRST (before vectorizing — avoids data leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# Now fit TF-IDF ONLY on training data, then transform both
vectorizer = TfidfVectorizer(max_features=3000)

X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocab + convert
X_test_tfidf  = vectorizer.transform(X_test)         # only convert (no learning)

print(f"\nTF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (test) : {X_test_tfidf.shape}")
print(f"\nVocabulary size: {len(vectorizer.vocabulary_)} words")
print("Sample words in vocabulary:", list(vectorizer.vocabulary_.keys())[:10])

# ── Step 6: Train Naive Bayes ───────────────────────────────────────────────
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("\nNaive Bayes — first 10 predictions vs actual:")
print("Predicted:", y_pred_nb[:10].tolist())
print("Actual   :", y_test[:10].tolist())

# ── Step 7: Train Decision Tree ─────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_tfidf, y_train)

y_pred_dt = dt_model.predict(X_test_tfidf)

# ── Step 8: Evaluate Both Models ────────────────────────────────────────────
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Naive Bayes ──
print("\n" + "="*50)
print("          NAIVE BAYES RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Ham', 'Spam']))

# ── Decision Tree ──
print("="*50)
print("          DECISION TREE RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['Ham', 'Spam']))

# ── Step 9: Visualizations ──────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Spam Detection — Model Analysis", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Label Distribution ──
ax1 = fig.add_subplot(gs[0, 0])
label_counts = df['label'].value_counts()
bars = ax1.bar(label_counts.index, label_counts.values,
               color=['steelblue', 'tomato'], edgecolor='black', width=0.5)
ax1.set_title("Dataset Label Distribution", fontweight='bold')
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             str(bar.get_height()), ha='center', va='bottom', fontweight='bold')

# ── 2. Message Length Distribution ──
ax2 = fig.add_subplot(gs[0, 1])
ham_lengths  = df[df['label'] == 'ham']['message'].str.len()
spam_lengths = df[df['label'] == 'spam']['message'].str.len()
ax2.hist(ham_lengths,  bins=40, alpha=0.6, color='steelblue', label='Ham',  edgecolor='white')
ax2.hist(spam_lengths, bins=40, alpha=0.6, color='tomato',    label='Spam', edgecolor='white')
ax2.set_title("Message Length Distribution", fontweight='bold')
ax2.set_xlabel("Character Count")
ax2.set_ylabel("Frequency")
ax2.legend()

# ── 3. Confusion Matrix — Naive Bayes ──
ax3 = fig.add_subplot(gs[1, 0])
cm_nb = confusion_matrix(y_test, y_pred_nb)
im3 = ax3.imshow(cm_nb, interpolation='nearest', cmap='Blues')
ax3.set_title("Confusion Matrix — Naive Bayes", fontweight='bold')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Ham', 'Spam']); ax3.set_yticklabels(['Ham', 'Spam'])
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm_nb[i, j]), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm_nb[i, j] > cm_nb.max() / 2 else 'black')

# ── 4. Confusion Matrix — Decision Tree ──
ax4 = fig.add_subplot(gs[1, 1])
cm_dt = confusion_matrix(y_test, y_pred_dt)
im4 = ax4.imshow(cm_dt, interpolation='nearest', cmap='Oranges')
ax4.set_title("Confusion Matrix — Decision Tree", fontweight='bold')
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Ham', 'Spam']); ax4.set_yticklabels(['Ham', 'Spam'])
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm_dt[i, j]), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm_dt[i, j] > cm_dt.max() / 2 else 'black')

# ── 5. Accuracy Comparison ──
ax5 = fig.add_subplot(gs[2, 0])
models     = ['Naive Bayes', 'Decision Tree']
accuracies = [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_dt)]
bars5 = ax5.bar(models, accuracies, color=['steelblue', 'darkorange'], edgecolor='black', width=0.4)
ax5.set_title("Model Accuracy Comparison", fontweight='bold')
ax5.set_ylabel("Accuracy")
ax5.set_ylim(0.9, 1.0)
for bar, acc in zip(bars5, accuracies):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f"{acc:.2%}", ha='center', va='bottom', fontweight='bold')

# ── 6. Top 15 Spam Words by TF-IDF Weight ──
ax6 = fig.add_subplot(gs[2, 1])
spam_indices  = np.where(y_train == 1)[0]
spam_tfidf    = X_train_tfidf[spam_indices]
mean_tfidf    = np.asarray(spam_tfidf.mean(axis=0)).flatten()
feature_names = np.array(vectorizer.get_feature_names_out())
top15_idx     = mean_tfidf.argsort()[-15:][::-1]
top15_words   = feature_names[top15_idx]
top15_scores  = mean_tfidf[top15_idx]
ax6.barh(top15_words[::-1], top15_scores[::-1], color='tomato', edgecolor='black')
ax6.set_title("Top 15 Spam Words (Avg TF-IDF)", fontweight='bold')
ax6.set_xlabel("Mean TF-IDF Score")

plt.savefig("spam_detection_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nVisualization saved to spam_detection_analysis.png")

# ── Step 10: Test with Custom Emails ────────────────────────────────────────

custom_emails = [
    "Congratulations! You've won a FREE $1,000 gift card. Click HERE to claim NOW!!!",
    "Hey, are we still meeting at 3pm tomorrow? Let me know!",
    "URGENT: Your account has been compromised. Call us immediately to claim your prize.",
    "Can you pick up some milk on your way home?",
    "FREE entry! Win a brand new iPhone. Text WIN to 87121 now!",
]

print("\n" + "="*50)
print("       CUSTOM EMAIL PREDICTIONS")
print("="*50)

for email in custom_emails:
    cleaned   = clean_text(email)
    vectorized = vectorizer.transform([cleaned])
    prediction = nb_model.predict(vectorized)[0]
    label      = "SPAM ***" if prediction == 1 else "HAM     "
    print(f"\n[{label}] {email[:60]}...")

# ── Step 11: Save Models ────────────────────────────────────────────────────
import joblib
import os

os.makedirs('models', exist_ok=True)

joblib.dump(nb_model,   'models/naive_bayes_spam.pkl')
joblib.dump(dt_model,   'models/decision_tree_spam.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\n" + "="*50)
print("         MODELS SAVED")
print("="*50)
print("✅ models/naive_bayes_spam.pkl")
print("✅ models/decision_tree_spam.pkl")
print("✅ models/tfidf_vectorizer.pkl")
print("\nTo load and use later:")
print("  nb  = joblib.load('models/naive_bayes_spam.pkl')")
print("  vec = joblib.load('models/tfidf_vectorizer.pkl')")
print("  prediction = nb.predict(vec.transform([clean_text(email)]))")
print("="*50)
