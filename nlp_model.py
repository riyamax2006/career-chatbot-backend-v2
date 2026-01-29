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

from dataset import CAREERS, SALARY_BOUNDS, TIME_HORIZON_MAP


class CareerNLPModel:
    """
    TF-IDF based NLP model for career matching.
    
    Converts career descriptions into TF-IDF vectors and computes
    cosine similarity with user query vectors.
    """
    
    def __init__(self):
        """Initialize and train the TF-IDF model on career corpus."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=500,
            min_df=1,
            max_df=0.95
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
                f"risk level {career['risk']}",
                f"{career['risk']} risk appetite tolerance",
            ]
            
            # Add keywords
            if 'keywords' in career:
                text_parts.extend(career['keywords'])
            
            # Add salary context
            salaries = career['salaries']
            text_parts.append(f"entry salary {salaries['entry']} LPA")
            text_parts.append(f"mid salary {salaries['mid']} LPA")
            text_parts.append(f"senior salary {salaries['senior']} LPA")
            
            # Categorize salary levels
            if salaries['entry'] <= 6:
                text_parts.append("entry level salary range")
            if 6 < salaries['mid'] <= 12:
                text_parts.append("growth salary range mid-range")
            if salaries['senior'] > 12:
                text_parts.append("premium salary high compensation")
            
            # Combine into single document
            document = ' '.join(text_parts)
            self.career_corpus.append(document)
    
    def _train_model(self):
        """Fit TF-IDF vectorizer on career corpus."""
        self.career_vectors = self.vectorizer.fit_transform(self.career_corpus)
        print(f"[NLP Model] Trained on {len(CAREERS)} careers")
        print(f"[NLP Model] Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def _build_user_query(self, salary_range: str, time_horizon: str, 
                          risk_appetite: str, skills: str = None) -> str:
        """
        Construct a text query from user inputs.
        
        This converts structured inputs into natural language that can be
        compared against career descriptions using TF-IDF similarity.
        
        Args:
            salary_range: 'entry' | 'growth' | 'premium'
            time_horizon: 'immediate' | 'mid_term' | 'long_term'
            risk_appetite: 'low' | 'medium' | 'high'
            skills: Optional free-text skills string
            
        Returns:
            Query string for TF-IDF matching
        """
        query_parts = []
        
        # Salary range context
        if salary_range == 'entry':
            query_parts.extend([
                "entry level salary range",
                "0 to 6 LPA",
                "starting career fresh graduate"
            ])
        elif salary_range == 'growth':
            query_parts.extend([
                "growth salary range mid-range",
                "6 to 12 LPA",
                "career growth progression"
            ])
        elif salary_range == 'premium':
            query_parts.extend([
                "premium salary high compensation",
                "above 12 LPA",
                "senior level executive high paying"
            ])
        
        # Time horizon context
        if time_horizon == 'immediate':
            query_parts.extend([
                "entry level entry salary",
                "immediate opportunity available now",
                "fresh graduate starting"
            ])
        elif time_horizon == 'mid_term':
            query_parts.extend([
                "mid level salary experience",
                "2 to 5 years career progression",
                "growth opportunity advancement"
            ])
        elif time_horizon == 'long_term':
            query_parts.extend([
                "senior level salary experience",
                "5 plus years career established",
                "senior executive leadership"
            ])
        
        # Risk appetite context
        if risk_appetite == 'low':
            query_parts.extend([
                "low risk stable secure",
                "job security predictable growth",
                "corporate government stable career"
            ])
        elif risk_appetite == 'medium':
            query_parts.extend([
                "medium risk balanced",
                "moderate stability growth",
                "balanced career opportunity"
            ])
        elif risk_appetite == 'high':
            query_parts.extend([
                "high risk appetite tolerance",
                "high reward potential startup",
                "entrepreneurial ambitious growth"
            ])
        
        # Add user-provided skills (most important for matching)
        if skills and skills.strip():
            query_parts.append(skills.strip())
        
        return ' '.join(query_parts)
    
    def compute_similarity(self, salary_range: str, time_horizon: str,
                          risk_appetite: str, skills: str = None) -> list:
        """
        Compute cosine similarity between user query and all careers.
        
        Args:
            salary_range: 'entry' | 'growth' | 'premium'
            time_horizon: 'immediate' | 'mid_term' | 'long_term'
            risk_appetite: 'low' | 'medium' | 'high'
            skills: Optional free-text skills string
            
        Returns:
            List of (career_index, similarity_score) tuples, sorted by score descending
        """
        # Build query from user inputs
        query = self._build_user_query(salary_range, time_horizon, risk_appetite, skills)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity with all careers
        similarities = cosine_similarity(query_vector, self.career_vectors).flatten()
        
        # Create (index, score) pairs and sort by score descending
        scored_indices = [(i, float(similarities[i])) for i in range(len(CAREERS))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        return scored_indices
    
    def get_recommendations(self, salary_range: str, time_horizon: str,
                           risk_appetite: str, skills: str = None,
                           top_k: int = 5) -> list:
        """
        Get top-k career recommendations based on NLP similarity.
        
        Args:
            salary_range: 'entry' | 'growth' | 'premium'
            time_horizon: 'immediate' | 'mid_term' | 'long_term'
            risk_appetite: 'low' | 'medium' | 'high'
            skills: Optional free-text skills string
            top_k: Number of recommendations to return
            
        Returns:
            List of dicts with career info and similarity scores
        """
        # Get similarity scores
        scored_indices = self.compute_similarity(
            salary_range, time_horizon, risk_appetite, skills
        )
        
        # Build results with career data
        results = []
        for idx, score in scored_indices[:top_k]:
            career = CAREERS[idx]
            results.append({
                'career': career,
                'similarity_score': score,
                'index': idx
            })
        
        return results
    
    def get_query_terms(self, salary_range: str, time_horizon: str,
                        risk_appetite: str, skills: str = None) -> dict:
        """
        Get the TF-IDF terms and weights for a user query.
        Useful for debugging and explainability.
        """
        query = self._build_user_query(salary_range, time_horizon, risk_appetite, skills)
        query_vector = self.vectorizer.transform([query])
        
        # Get feature names and their weights
        feature_names = self.vectorizer.get_feature_names_out()
        weights = query_vector.toarray().flatten()
        
        # Get non-zero terms
        non_zero = [(feature_names[i], weights[i]) for i in range(len(weights)) if weights[i] > 0]
        non_zero.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'query_text': query,
            'top_terms': non_zero[:20]
        }


# Singleton instance - initialized once at module load
_nlp_model = None

def get_nlp_model() -> CareerNLPModel:
    """Get or create the singleton NLP model instance."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = CareerNLPModel()
    return _nlp_model
