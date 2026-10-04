"""
Flask Application for Fake News Detection.
Serves prediction endpoints, NLP feature extraction, model explainability, and static web UI.
"""

import os
import json
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from nlp_analyzer import NLPAnalyzer
from model.train_model import train_fake_news_model

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

MODEL_PATH = os.path.join("model", "fake_news_model.joblib")
VECTORIZER_PATH = os.path.join("model", "vectorizer.joblib")
METRICS_PATH = os.path.join("model", "model_metrics.json")

# Initialize NLP Analyzer
analyzer = NLPAnalyzer()

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_or_train_model():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model checkpoint missing. Running model training...")
        train_fake_news_model()

    print("Loading machine learning model and TF-IDF vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model loaded successfully!")

# Ensure model loaded on startup
load_or_train_model()

# --- Sample Articles for 1-Click Testing ---
SAMPLE_ARTICLES = [
    {
        "id": "sample-1",
        "category": "Real Science",
        "badge_color": "emerald",
        "title": "NASA James Webb Space Telescope Observes Oldest Galaxy Clusters in Known Universe",
        "text": "Astronomers using NASA's James Webb Space Telescope have discovered some of the earliest galaxy clusters ever observed, dating back over 13 billion years. The discovery offers crucial insights into cosmic structure formation shortly after the Big Bang. According to lead researchers at the Space Telescope Science Institute in Baltimore, spectroscopic analysis confirmed the red-shift values of these distant stellar formations. The findings have been peer-reviewed and published in the Astronomical Journal."
    },
    {
        "id": "sample-2",
        "category": "Fake Conspiracy",
        "badge_color": "rose",
        "title": "SHOCKING PROOF: Secret Government Satellite Beam Mind Control Waves Detected Across Major Cities!",
        "text": "Secret whistleblowers inside top intelligence agencies have finally leaked classified documents proving government satellites are broadcasting subliminal electromagnetic waves! They don't want you to know the truth! Millions of unsuspecting citizens are being manipulated right now to control their thoughts and buy specific products. Mainstream media is hiding this massive cover-up! Share this urgent report before it gets censored and deleted from the internet!"
    },
    {
        "id": "sample-3",
        "category": "Fake Miracle Cure",
        "badge_color": "amber",
        "title": "Doctors Are Furious! Ancient Miracle Fruit Cures All Diseases in 24 Hours, Big Pharma Panics!",
        "text": "Big Pharma is trying to ban this unbelievable natural secret! A mysterious Amazonian berry discovered by rogue researchers instantly eradicates cancer, diabetes, and heart disease overnight! Corporate drug executives are spending millions to keep this magical cure hidden from the public so they can keep profiting off expensive pills. Don't fall for their lies! Click here now to claim your secret stash before supplies run out forever!"
    },
    {
        "id": "sample-4",
        "category": "Real Economics",
        "badge_color": "emerald",
        "title": "Federal Reserve Maintains Interest Rates Amid Moderating Inflation Data",
        "text": "The Federal Reserve voted unanimously to keep benchmark interest rates unchanged at its monthly policy meeting. Federal Reserve Chairman announced that recent economic indicators suggest steady job creation alongside gradually cooling inflation numbers. Treasury yields responded with modest declines across major indices. Economists noted that the central bank remains data-dependent before considering any future monetary easing."
    },
    {
        "id": "sample-5",
        "category": "Real Tech & AI",
        "badge_color": "emerald",
        "title": "European Union Reaches Landmark Agreement on Artificial Intelligence Regulation",
        "text": "European Union lawmakers successfully concluded negotiations on comprehensive legislation governing artificial intelligence systems across member nations. The regulations categorize AI applications based on risk levels, establishing strict compliance guidelines for high-risk biometric and algorithmic models. Tech Industry analysts expect the framework to set global compliance standards for AI safety and data privacy transparency."
    }
]

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()

        if not text and not title:
            return jsonify({"error": "Please provide an article title or body text for analysis."}), 400

        full_content = f"{title} {text}".strip()
        cleaned_content = analyzer.clean_text(full_content)

        # 1. Scikit-learn TF-IDF Vectorization & Prediction
        tfidf_vec = vectorizer.transform([cleaned_content])
        probabilities = model.predict_proba(tfidf_vec)[0]
        prediction_label = int(model.predict(tfidf_vec)[0])

        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])

        # Confidence percentage
        is_real = prediction_label == 1
        confidence = round((real_prob if is_real else fake_prob) * 100, 1)

        # 2. NLP Feature Extraction
        linguistic_metrics = analyzer.analyze_linguistic_features(title, text)

        # Adjust score if heavy clickbait / sensational patterns override neutral vocabulary
        sensationalism = linguistic_metrics["sensationalism_score"]
        clickbait = linguistic_metrics["clickbait_score"]
        
        # Fine-tune probability with NLP linguistic heuristics for robust out-of-distribution detection
        nlp_fake_penalty = 0.0
        if sensationalism > 50:
            nlp_fake_penalty += (sensationalism - 50) * 0.005
        if clickbait > 50:
            nlp_fake_penalty += (clickbait - 50) * 0.004

        adjusted_fake_prob = min(0.99, fake_prob + nlp_fake_penalty)
        adjusted_real_prob = max(0.01, 1.0 - adjusted_fake_prob)

        final_label = "Real" if adjusted_real_prob >= 0.5 else "Fake"
        final_confidence = round((adjusted_real_prob if final_label == "Real" else adjusted_fake_prob) * 100, 1)

        # 3. Model Explainability: Extract key word influences
        influencers = analyzer.extract_key_influencers(cleaned_content, vectorizer, model)

        # 4. Highlighted Text Segments
        highlighted_segments = analyzer.generate_highlighted_text(text if text else title, influencers["fake_indicators"])

        response = {
            "prediction": final_label,
            "status": "Likely Real" if final_label == "Real" else "Likely Fake",
            "confidence": final_confidence,
            "probabilities": {
                "real": round(adjusted_real_prob * 100, 1),
                "fake": round(adjusted_fake_prob * 100, 1)
            },
            "linguistic_analysis": linguistic_metrics,
            "key_influencers": influencers,
            "highlighted_segments": highlighted_segments,
            "summary": {
                "word_count": linguistic_metrics["total_word_count"],
                "sensationalism_level": "High" if sensationalism > 60 else ("Medium" if sensationalism > 30 else "Low"),
                "clickbait_risk": "High" if clickbait > 60 else ("Medium" if clickbait > 30 else "Low"),
                "subjectivity_level": "High" if linguistic_metrics["subjectivity_score"] > 60 else ("Moderate" if linguistic_metrics["subjectivity_score"] > 40 else "Low / Objective")
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the article: {str(e)}"}), 500

@app.route('/api/samples', methods=['GET'])
def get_samples():
    return jsonify({"samples": SAMPLE_ARTICLES})

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    else:
        return jsonify({"error": "Metrics file not found"}), 444

if __name__ == '__main__':
    print("\n========================================================")
    print(" Fake News Detection Web App Server")
    print(" Running on http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
