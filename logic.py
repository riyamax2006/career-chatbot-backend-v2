"""
Business Logic for Career Recommendations.

This module handles:
1. Input validation
2. Combining NLP similarity with business constraints
3. Post-filtering and ranking
4. Feasibility notes generation

The NLP model provides similarity scores, this module applies
additional business rules for final recommendations.
"""

from dataset import (
    CAREERS, 
    VALID_SALARY_RANGES, 
    VALID_TIME_HORIZONS, 
    VALID_RISK_LEVELS,
    SALARY_BOUNDS,
    TIME_HORIZON_MAP
)
from nlp_model import get_nlp_model


class ValidationError(Exception):
    """Raised when input validation fails."""
    def __init__(self, message, field=None):
        self.message = message
        self.field = field
        super().__init__(self.message)


def validate_inputs(data: dict) -> dict:
    """
    Validate and normalize all input parameters.
    
    Args:
        data: Dict with salary_range, time_horizon, risk_appetite, skills (optional)
        
    Returns:
        Dict with validated and normalized values
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    
    # Validate salary_range
    salary_range = data.get("salary_range")
    if salary_range is None:
        raise ValidationError("salary_range is required", field="salary_range")
    salary_range = str(salary_range).lower().strip()
    if salary_range not in VALID_SALARY_RANGES:
        raise ValidationError(
            f"Invalid salary_range '{salary_range}'. Must be one of: {', '.join(VALID_SALARY_RANGES)}",
            field="salary_range"
        )
    
    # Validate time_horizon  
    time_horizon = data.get("time_horizon")
    if time_horizon is None:
        raise ValidationError("time_horizon is required", field="time_horizon")
    time_horizon = str(time_horizon).lower().strip()
    if time_horizon not in VALID_TIME_HORIZONS:
        raise ValidationError(
            f"Invalid time_horizon '{time_horizon}'. Must be one of: {', '.join(VALID_TIME_HORIZONS)}",
            field="time_horizon"
        )
    
    # Validate risk_appetite
    risk_appetite = data.get("risk_appetite")
    if risk_appetite is None:
        raise ValidationError("risk_appetite is required", field="risk_appetite")
    risk_appetite = str(risk_appetite).lower().strip()
    if risk_appetite not in VALID_RISK_LEVELS:
        raise ValidationError(
            f"Invalid risk_appetite '{risk_appetite}'. Must be one of: {', '.join(VALID_RISK_LEVELS)}",
            field="risk_appetite"
        )
    
    # Skills is optional
    skills = data.get("skills", "")
    if skills:
        skills = str(skills).strip()
    
    return {
        "salary_range": salary_range,
        "time_horizon": time_horizon,
        "risk_appetite": risk_appetite,
        "skills": skills
    }


def get_salary_for_horizon(career: dict, time_horizon: str) -> float:
    """Get the salary value for a career at a given time horizon."""
    horizon_key = TIME_HORIZON_MAP.get(time_horizon, "entry")
    return career["salaries"].get(horizon_key, 0)


def calculate_salary_score(career: dict, salary_range: str, time_horizon: str) -> float:
    """
    Calculate a soft score (0.0 - 1.0) for salary fit.
    
    Returns:
        Float score between 0.0 and 1.0
    """
    salary = get_salary_for_horizon(career, time_horizon)
    min_sal, max_sal = SALARY_BOUNDS.get(salary_range, (0, float("inf")))
    
    # Perfect match: In range
    if min_sal <= salary <= max_sal or (max_sal == float("inf") and salary >= min_sal):
        return 1.0
    
    # Above range: Good but slightly penalized (0.8) to prefer exact matches
    if salary > max_sal and max_sal != float("inf"):
        return 0.8
    
    # Below range: Decay based on how far off it is
    # Score = 1.0 - 0.2 * (missing_ratio)
    if min_sal > 0:
        ratio = salary / min_sal
        # Linear decay mapping 1.0 -> 0.0 to 0.8 -> 0.2
        return max(0.2, 0.2 + (0.6 * ratio))
        
    return 0.2


def calculate_risk_score(career: dict, risk_appetite: str) -> float:
    """
    Calculate a soft score (0.0 - 1.0) for risk fit.
    
    Returns:
        Float score between 0.0 and 1.0
    """
    career_risk = career["risk"].lower()
    risk_levels = ["low", "medium", "high"]
    
    try:
        career_idx = risk_levels.index(career_risk)
        user_idx = risk_levels.index(risk_appetite)
        
        # User's appetite is a ceiling
        if career_idx <= user_idx:
            return 1.0  # Perfect fit (within tolerance)
        
        # Career is riskier than user wants
        # Gap of 1 (Low user vs Med career) -> 0.7
        # Gap of 2 (Low user vs High career) -> 0.4
        gap = career_idx - user_idx
        return max(0.2, 1.0 - (gap * 0.3))
        
    except ValueError:
        return 0.5


def get_recommendations(salary_range: str, time_horizon: str, 
                        risk_appetite: str, skills: str = None) -> tuple:
    """
    Get career recommendations using Weighted Additive Scoring.
    
    Formula:
    Score = (W_nlp * NLP_Score) + (W_salary * Salary_Score) + (W_risk * Risk_Score)
    
    Args:
        salary_range: 'entry' | 'growth' | 'premium'
        time_horizon: 'immediate' | 'mid_term' | 'long_term'
        risk_appetite: 'low' | 'medium' | 'high'
        skills: Optional free-text skills
        
    Returns:
        Tuple of (recommended_careers, feasibility_note)
    """
    # 1. Get NLP scores for ALL careers
    nlp_model = get_nlp_model()
    # top_k=None returns all
    nlp_results = nlp_model.get_recommendations(
        salary_range, time_horizon, risk_appetite, skills, top_k=None
    )
    
    # 2. Weights Configuration
    W_NLP = 0.4
    W_SALARY = 0.3
    W_RISK = 0.3
    
    scored_careers = []
    
    for result in nlp_results:
        career = result['career']
        nlp_score = result['similarity_score']
        
        # Calculate Component Scores (0.0 - 1.0)
        salary_score = calculate_salary_score(career, salary_range, time_horizon)
        risk_score = calculate_risk_score(career, risk_appetite)
        
        # Calculate Weighted Total
        total_score = (W_NLP * nlp_score) + (W_SALARY * salary_score) + (W_RISK * risk_score)
        
        scored_careers.append({
            'career': career,
            'role': career['role'],
            'total_score': total_score,
            'nlp_score': nlp_score,
            'salary_score': salary_score,
            'risk_score': risk_score
        })
    
    # 3. Sort by Total Score
    scored_careers.sort(key=lambda x: x['total_score'], reverse=True)
    
    # 4. Format Output
    recommended = []
    horizon_key = TIME_HORIZON_MAP.get(time_horizon, "entry")
    
    # We take the top 3 matches
    # But we want to ensure they aren't complete garbage matches (e.g. score < 0.4)
    # However, since we must return something, we just return top 3.
    
    for item in scored_careers[:3]:
        career = item['career']
        salary = career['salaries'].get(horizon_key, 0)
        
        # Build explanation
        # "High match (85%). Matches salary expectations. Slightly higher risk than requested."
        reasons = []
        
        # NLP Context
        if item['nlp_score'] > 0.3:
            reasons.append(f"role match ({item['nlp_score']:.0%}).")
        else:
            reasons.append("Potential role match.")
            
        # Salary Context
        if item['salary_score'] == 1.0:
            reasons.append(f"Fits salary range ({salary} LPA).")
        elif item['salary_score'] == 0.8:
            reasons.append(f"Exceeds salary target ({salary} LPA).")
        else:
            reasons.append(f"Salary: {salary} LPA.")
            
        # Risk Context
        if item['risk_score'] < 1.0:
            reasons.append("Higher risk profile.")
            
        reason_str = " ".join(reasons)
        
        recommended.append({
            "role": career['role'],
            "reason": f"Match Score: {item['total_score']:.0%}. {reason_str}"
        })
    
    # 5. Generate Feasibility Note
    feasibility_note = generate_feasibility_note(
        scored_careers, salary_range, time_horizon, risk_appetite
    )
    
    return recommended, feasibility_note


def generate_feasibility_note(scored_careers: list, salary_range: str,
                              time_horizon: str, risk_appetite: str) -> str:
    """Generate helpful feasibility note based on top scores."""
    
    top_score = scored_careers[0]['total_score'] if scored_careers else 0
    
    if top_score > 0.85:
        return "Excellent matches found aligned with your profile and preferences."
    elif top_score > 0.70:
        return "Good matches found. Some trade-offs in salary or risk may be present."
    elif top_score > 0.50:
        return "Moderate matches. Recommendations balance your constraints with available options."
    else:
        # Low scores - give specific advice
        notes = []
        if salary_range == "premium" and risk_appetite == "low":
            notes.append("High salary with low risk is rare.")
        if salary_range == "premium" and time_horizon == "immediate":
            notes.append("Starting directly at premium salary is difficult.")
            
        if not notes:
            notes.append("Your criteria are very restrictive.")
            
        return f"Limited strong matches (Top score: {top_score:.0%}). { ' '.join(notes) } Consider broadening your search."
