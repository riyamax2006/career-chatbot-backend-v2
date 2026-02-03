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
from dataset import get_dataset_vocabulary



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
        self.dataset_vocab = get_dataset_vocabulary()

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
    


    def _expand_synonyms(self, text: str) -> str:
        intent_map = {
            "treat": ["patient", "care", "medical"],
            "diagnose": ["diagnosis", "medical", "clinical"],
            "heal": ["treatment", "care"],
            "illness": ["disease", "medical", "health"],
    
            "stock": ["finance", "investment", "market"],
            "stocks": ["finance", "investment", "market"],
            "profit": ["revenue", "earnings", "finance"],
            "trading": ["investment", "market"],
    
            "business": ["management", "strategy"],
            "market": ["finance", "business"],
    
            "code": ["software", "programming", "development"],
            "coding": ["software", "programming"],
            "ai": ["machine", "learning", "data"],
            "data": ["analytics", "analysis"],
        }
    
        words = text.lower().split()
        expanded = list(words)
    
        for word in words:
            if word in intent_map:
                for candidate in intent_map[word]:
                    if candidate in self.dataset_vocab:
                        expanded.append(candidate)
    
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
    
        # DO NOT clean text — keep full intent
        expanded_skills = self._expand_synonyms(skills.lower())
    
        # Repeat to strengthen TF-IDF signal
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
