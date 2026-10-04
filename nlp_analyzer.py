"""
NLP Feature Extractor and Model Explainability Engine for Fake News Detection.
"""

import re
import math
import numpy as np

# List of common sensationalist, clickbait, and conspiracy trigger words
SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "exposed", "bombshell", "leaked", "secret", "whistleblower",
    "furious", "miracle", "panic", "banned", "censored", "they don't want you to know",
    "hidden truth", "mind control", "aliens", "ufo", "conspiracy", "bizarre", "mind-blowing",
    "doctors are stunned", "overnight", "instant", "guaranteed", "proven secret", "urgent alert",
    "patriots", "shadow government", "deep state", "cover-up", "terrified", "baffled",
    "you won't believe", "what happened next", "click here", "share before deleted"
}

CLICKBAIT_PATTERNS = [
    r"\b\d+\s+(reasons|ways|things|secrets|tricks|facts)\b",
    r"\bthis is why\b",
    r"\byou won't believe\b",
    r"\bwhat happens next\b",
    r"\bshocking proof\b",
    r"\bdoctors (hate|furious|stunned)\b",
    r"\bsecret (revealed|exposed|leaked)\b",
    r"\bdon't want you to know\b",
    r"\bnever before seen\b",
    r"\bviral video\b"
]

class NLPAnalyzer:
    def __init__(self):
        pass

    def clean_text(self, text):
        """Standardize and clean raw text for processing."""
        if not text:
            return ""
        text = str(text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_linguistic_features(self, title, text):
        """
        Extract linguistic metrics:
        - Sensationalism score (0-100)
        - Clickbait score (0-100)
        - Uppercase ratio
        - Exclamation mark count
        - Lexical diversity (unique words / total words)
        - Subjectivity score (0-100)
        """
        full_text = f"{title} {text}"
        words = re.findall(r'\b[a-zA-Z]{2,}\b', full_text.lower())
        total_words = len(words) if len(words) > 0 else 1

        # 1. Sensationalism Score
        sensational_count = sum(1 for w in words if w in SENSATIONAL_WORDS)
        for phrase in ["mind control", "deep state", "shadow government", "hidden truth", "doctors are furious"]:
            if phrase in full_text.lower():
                sensational_count += 3

        exclamations = full_text.count("!")
        uppercase_words = sum(1 for w in re.findall(r'\b[A-Z]{2,}\b', full_text) if w not in ["NASA", "WHO", "FDA", "UN", "EU", "MIT", "USA", "OECD", "IQ", "CBDC", "GPS", "IT", "EV", "AI", "TV", "3D"])
        
        sensational_raw = (sensational_count * 15) + (exclamations * 8) + (uppercase_words * 5)
        sensationalism_score = min(100, max(0, round(sensational_raw)))

        # 2. Clickbait Score
        clickbait_hits = 0
        title_lower = title.lower() if title else ""
        for pattern in CLICKBAIT_PATTERNS:
            if re.search(pattern, title_lower):
                clickbait_hits += 1

        if any(w in title_lower for w in ["shocking", "unbelievable", "exposed", "leaked", "secret", "revealed"]):
            clickbait_hits += 1

        if title and title.isupper():
            clickbait_hits += 2

        clickbait_score = min(100, round((clickbait_hits / 3.0) * 100))

        # 3. Lexical Diversity
        unique_words = len(set(words))
        lexical_diversity = round((unique_words / total_words) * 100, 1)

        # 4. Subjectivity Score (Heuristic NLP estimation)
        subjective_indicators = {
            "think", "believe", "feel", "claim", "allegedly", "shocking", "horrible",
            "wonderful", "terrible", "amazing", "unbelievable", "disastrous", "outrageous",
            "insane", "crazy", "secret", "miracle", "furious", "baffled", "heroic"
        }
        objective_indicators = {
            "according", "published", "study", "data", "report", "percent", "university",
            "journal", "researchers", "officials", "confirmed", "measured", "analysis",
            "dr.", "department", "institutes", "spokesperson", "agency", "finding"
        }

        subj_count = sum(1 for w in words if w in subjective_indicators)
        obj_count = sum(1 for w in words if w in objective_indicators)
        
        raw_subj = 50 + (subj_count * 8) - (obj_count * 6)
        subjectivity_score = min(100, max(0, round(raw_subj)))

        return {
            "sensationalism_score": sensationalism_score,
            "clickbait_score": clickbait_score,
            "uppercase_words": uppercase_words,
            "exclamation_count": exclamations,
            "total_word_count": total_words,
            "lexical_diversity": lexical_diversity,
            "subjectivity_score": subjectivity_score
        }

    def extract_key_influencers(self, text, vectorizer, model, top_n=8):
        """
        Explainability engine: inspects TF-IDF feature weights for terms in input text.
        Returns top features pushing toward Fake (label 0) and Real (label 1).
        """
        if not hasattr(model, "coef_") and hasattr(model, "calibrated_classifiers_"):
            # If using CalibratedClassifierCV, extract underlying estimator
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model

        if not hasattr(base_model, "coef_"):
            return {"fake_indicators": [], "real_indicators": []}

        coefs = base_model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        # Transform single document
        tfidf_vec = vectorizer.transform([text])
        non_zero_indices = tfidf_vec.nonzero()[1]

        fake_indicators = []
        real_indicators = []

        for idx in non_zero_indices:
            feature_name = feature_names[idx]
            weight = coefs[idx] * tfidf_vec[0, idx]
            
            # Weight > 0 pushing toward Real (label 1 in our dataset), Weight < 0 pushing toward Fake
            if weight > 0:
                real_indicators.append({
                    "term": feature_name,
                    "weight": round(float(weight), 4),
                    "impact": "Real"
                })
            elif weight < 0:
                fake_indicators.append({
                    "term": feature_name,
                    "weight": round(float(abs(weight)), 4),
                    "impact": "Fake"
                })

        # Sort by weight magnitude descending
        fake_indicators = sorted(fake_indicators, key=lambda x: x["weight"], reverse=True)[:top_n]
        real_indicators = sorted(real_indicators, key=lambda x: x["weight"], reverse=True)[:top_n]

        return {
            "fake_indicators": fake_indicators,
            "real_indicators": real_indicators
        }

    def generate_highlighted_text(self, text, fake_terms):
        """
        Marks terms in the text that are sensational or flagged by fake news features.
        Returns list of text segments with flag status.
        """
        if not text:
            return []

        flagged_terms = set(SENSATIONAL_WORDS)
        for item in fake_terms:
            flagged_terms.add(item["term"].lower())

        # Build regex for matching flagged terms
        pattern = r'\b(' + '|'.join(re.escape(term) for term in sorted(flagged_terms, key=len, reverse=True) if len(term) > 2) + r')\b'
        
        segments = []
        last_idx = 0

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            if start > last_idx:
                segments.append({"text": text[last_idx:start], "flagged": False})
            segments.append({"text": text[start:end], "flagged": True})
            last_idx = end

        if last_idx < len(text):
            segments.append({"text": text[last_idx:], "flagged": False})

        return segments
