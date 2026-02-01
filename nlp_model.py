"""
NLP Model for Career Recommendations using TF-IDF.

This module implements:
1. TF-IDF vectorization of career descriptions
2. User query construction from structured inputs
3. Cosine similarity computation for ranking careers

NO rule-based matching - purely NLP-driven recommendations.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from dataset import CAREERS

# Terms to explicitly REMOVE from NLP query to prevent rules from polluting semantic matching
IGNORE_TERMS = {
    # Salary terms
    "salary", "lpa", "compensation", "money", "pay", "earning", "entry", "growth", "premium", "rich", "wealth",
    "high paying", "low paying", "0 to 6", "6 to 12", "above 12",
    # Risk terms
    "risk", "stable", "secure", "security", "startup", "corporate", "government", "entrepreneurial", "predictable",
    "balanced", "volatile", "safe", "chance",
    # Timeline terms
    "immediate", "urgent", "start", "now", "years", "long term", "mid term", "future", "career path",
    "progression", "experience", "seniority", "junior", "mid level", "senior level"
}

class CareerNLPModel:
    """
    TF-IDF based NLP model for career matching.
    """
    
    def __init__(self):
        """Initialize and train the TF-IDF model on career corpus."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=None,   # Use all features
            min_df=1,
            max_df=1.0  # Allow terms that appear in all documents if corpus is small
        )
        self.career_vectors = None
        self.career_corpus = None
        self._build_corpus()
        self._train_model()
    
    def _build_corpus(self):
        """
        Build text corpus from career data.
        Each career becomes a rich text document combining all its attributes.
        """
        self.career_corpus = []
        
        for career in CAREERS:
            # Combine all text attributes into one document
            text_parts = [
                career['role'],
                career['category'],
                career['description'],
            ]
            
            # Add keywords (BOOSTED 5x to ensure strong matching)
            if 'keywords' in career:
                # Repeat keywords multiple times to increase TF (Term Frequency)
                # This ensures that if a user searches for a keyword, it ranks highly.
                boosted_keywords = (career['keywords'] * 5)
                text_parts.extend(boosted_keywords)
            
            # Combine into single document
            document = ' '.join(text_parts)
            self.career_corpus.append(document)
    
    def _train_model(self):
        """Fit TF-IDF vectorizer on career corpus."""
        self.career_vectors = self.vectorizer.fit_transform(self.career_corpus)
        print(f"[NLP Model] Trained on {len(CAREERS)} careers")
    
    def _clean_text(self, text: str) -> str:
        """Remove rule-based terms from text."""
        if not text:
            return ""
        
        # Tokenize and filter
        words = text.lower().split()
        cleaned = [w for w in words if w not in IGNORE_TERMS]
        
        # Also remove multi-word phrases (naive approach)
        text_clean = " ".join(cleaned)
        for term in IGNORE_TERMS:
            if " " in term: # If it's a phrase in ignore list
                text_clean = text_clean.replace(term, "")
                
        return text_clean.strip()

    def _expand_synonyms(self, text: str) -> str:
        """Expand domain terms with synonyms to improve matching."""
        # Simple synonym mapping for common career domains
        synonyms = {
            "healthcare": "medical doctor hospital health physician nurse",
            "medical": "healthcare doctor hospital health",
            "tech": "technology software engineering developer data ai",
            "programming": "coding software development engineering",
            "ai": "artificial intelligence machine learning data science",
            "finance": "banking investment money accounting",
            "business": "management corporate strategy consulting",
        }
        
        words = text.lower().split()
        expanded = list(words)
        
        for word in words:
            if word in synonyms:
                expanded.append(synonyms[word])
                
        return " ".join(expanded)

    def _build_user_query(self, skills: str = "") -> str:
        """
        Construct a text query from user inputs.
        
        Args:
            skills: User provided skills/intent
            
        Returns:
            Query string for TF-IDF matching
        """
        if not skills:
            return ""

        # 1. Clean the input (remove salary/risk/timeline terms)
        clean_skills = self._clean_text(skills)
        
        if not clean_skills:
            return ""

        # 2. Expand Synonyms
        expanded_skills = self._expand_synonyms(clean_skills)

        # 3. Construct Query: Skills repeated 3 times
        # Repetition emphasizes these terms over any background noise
        query = f"{expanded_skills} {expanded_skills} {expanded_skills}"
        
        return query
    
    def get_recommendations(self, skills: str = "", top_k: int = None) -> list:
        """
        Get career recommendations based on NLP similarity.
        
        Args:
            skills: Free-text skills string
            top_k: Number of recommendations to return (None = all)
            
        Returns:
            List of dicts with career info and similarity scores
        """
        query = self._build_user_query(skills)
        
        if not query:
            # Return empty or handling? 
            # If no query, cosine similarity will be 0
            # We return all with 0 score
            results = []
            for idx, career in enumerate(CAREERS):
                results.append({
                    'career': career,
                    'similarity_score': 0.0,
                    'index': idx
                })
            return results

        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.career_vectors).flatten()
        
        # Build results
        results = []
        scored_indices = [(i, float(similarities[i])) for i in range(len(CAREERS))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        limit = top_k if top_k is not None else len(scored_indices)
        
        for idx, score in scored_indices[:limit]:
            career = CAREERS[idx]
            results.append({
                'career': career,
                'similarity_score': score,
                'index': idx
            })
        
        return results

# Singleton instance
_nlp_model = None

def get_nlp_model() -> CareerNLPModel:
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = CareerNLPModel()
    return _nlp_model
