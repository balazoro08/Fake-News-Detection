# 🛡️ VeritasAI - Fake News Detection & NLP Intelligence Platform

A high-performance, explainable **Fake News Detection Web Application** built with **Python**, **Scikit-Learn**, **Natural Language Processing (NLP)**, **Flask**, and a modern glassmorphic Web UI.

![VeritasAI Platform Banner](https://img.shields.io/badge/Python-3.13-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![Flask API](https://img.shields.io/badge/Flask-3.0+-green.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%26%20Linguistic%20Analysis-indigo.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

---

## 🌟 Key Features

- 🤖 **Machine Learning Classification**: Predicts whether a news article is **Likely Real** or **Likely Fake** with exact confidence percentage scoring.
- 📊 **Model Explainability (TF-IDF Feature Contributions)**: Extracts the top n-gram keywords and phrases in the article that influenced the classification towards Real or Fake.
- 🔍 **Interactive Text Highlighter**: Highlights clickbait, sensational, or high-risk terms directly within the article text.
- 📈 **NLP Linguistic Metrics**:
  - **Sensationalism Index (0-100%)**: Detects emotional trigger terms, ALL-CAPS density, and dramatic punctuation.
  - **Clickbait Risk Meter (0-100%)**: Identifies curiosity gaps, exaggerated promises, and numerical hooks.
  - **Subjectivity & Tone Metric**: Distinguishes between subjective reporting and objective factual coverage.
  - **Lexical Diversity**: Measures unique word density.
- ⚡ **1-Click Test Samples**: Pre-loaded real and fake sample articles covering space discovery, miracle cures, economic policy, AI regulation, and conspiracy hoaxes.
- 📜 **Session Analysis History**: Tracks and lets you re-analyze previously checked articles during your browser session.
- 🎨 **Glassmorphic Dark Mode UI**: Built with modern CSS design system, animated glowing background elements, dynamic gauges, and responsive layouts.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13, Flask, Flask-CORS
- **Machine Learning**: Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`)
- **NLP & Feature Extraction**: Custom N-Gram & linguistic feature engineering engine
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism, HSL Design Tokens), JavaScript (ES6+)
- **Serialization**: Joblib, JSON

---

## 📁 Repository Structure

```
Fake-News-Detection/
├── data/
│   ├── create_dataset.py       # Script to build multi-domain dataset
│   └── news_dataset.csv        # Real & Fake news annotated dataset
├── model/
│   ├── train_model.py          # Scikit-learn ML training pipeline
│   ├── fake_news_model.joblib  # Trained model weights
│   ├── vectorizer.joblib       # Saved TF-IDF vectorizer
│   └── model_metrics.json      # Model performance metrics (Accuracy, F1, CM)
├── static/
│   ├── index.html              # Modern Web Application UI
│   ├── css/
│   │   └── style.css           # Glassmorphic CSS styling system
│   └── js/
│       └── main.js             # Frontend interactivity & API consumer
├── nlp_analyzer.py             # NLP features, sentiment & explainability engine
├── app.py                      # Flask API web server
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---



---

## 🎯 Example Model Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | `90.0%` |
| **Precision** | `0.857` |
| **Recall** | `1.000` |
| **F1 Score** | `0.923` |
| **Feature Vocab Size** | `8,000 N-Grams` |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
