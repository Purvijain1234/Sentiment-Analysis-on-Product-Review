from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ── Load saved model & vectorizer ──────────────────────────────────────────────
# Make sure these pkl files are in the same folder as app.py
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# ── Preprocessing (same as training) ──────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'Empty review'}), 400

    cleaned    = preprocess(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # If model supports predict_proba (e.g. Logistic Regression)
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(vectorized)[0]
        classes    = list(model.classes_)
        confidence = round(float(max(proba)) * 100, 1)
        proba_dict = {cls: round(float(p)*100,1) for cls, p in zip(classes, proba)}
    else:
        proba_dict = {}

    return jsonify({
        'sentiment':  prediction,
        'confidence': confidence,
        'proba':      proba_dict
    })

if __name__ == '__main__':
    app.run(debug=True)
