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


def check_salary_constraint(career: dict, salary_range: str, time_horizon: str) -> tuple:
    """
    Check if career's salary matches user's target range.
    
    Returns:
        (matches: bool, multiplier: float)
        multiplier is used to boost/penalize score based on fit
    """
    salary = get_salary_for_horizon(career, time_horizon)
    min_sal, max_sal = SALARY_BOUNDS.get(salary_range, (0, float("inf")))
    
    if min_sal <= salary <= max_sal or (max_sal == float("inf") and salary >= min_sal):
        return True, 1.0  # Perfect match
    
    # Check how close it is
    if salary < min_sal:
        # Below range - penalize based on gap
        gap_ratio = salary / min_sal if min_sal > 0 else 0
        return False, max(0.3, gap_ratio)
    else:
        # Above range - slight boost (user might like higher salary)
        return False, 0.8


def check_risk_constraint(career: dict, risk_appetite: str) -> tuple:
    """
    Check if career's risk matches user's appetite.
    
    Returns:
        (matches: bool, multiplier: float)
    """
    career_risk = career["risk"].lower()
    risk_levels = ["low", "medium", "high"]
    
    try:
        career_idx = risk_levels.index(career_risk)
        user_idx = risk_levels.index(risk_appetite)
        
        # User's appetite is a ceiling
        if career_idx <= user_idx:
            return True, 1.0  # Within tolerance
        else:
            # Career is riskier than user wants
            gap = career_idx - user_idx
            return False, max(0.4, 1.0 - (gap * 0.3))
    except ValueError:
        return False, 0.5


def get_recommendations(salary_range: str, time_horizon: str, 
                        risk_appetite: str, skills: str = None) -> tuple:
    """
    Get career recommendations using NLP similarity + business constraints.
    
    This is the main entry point that combines:
    1. TF-IDF similarity scores from NLP model
    2. Salary range constraints
    3. Risk tolerance matching
    
    Args:
        salary_range: 'entry' | 'growth' | 'premium'
        time_horizon: 'immediate' | 'mid_term' | 'long_term'
        risk_appetite: 'low' | 'medium' | 'high'
        skills: Optional free-text skills
        
    Returns:
        Tuple of (recommended_careers, feasibility_note)
    """
    # Get NLP model and compute similarity scores
    nlp_model = get_nlp_model()
    nlp_results = nlp_model.get_recommendations(
        salary_range, time_horizon, risk_appetite, skills, top_k=16
    )
    
    # Apply business constraints to NLP results
    scored_careers = []
    
    for result in nlp_results:
        career = result['career']
        nlp_score = result['similarity_score']
        
        # Apply constraint multipliers
        salary_ok, salary_mult = check_salary_constraint(career, salary_range, time_horizon)
        risk_ok, risk_mult = check_risk_constraint(career, risk_appetite)
        
        # Combined score: NLP similarity * constraint multipliers
        final_score = nlp_score * salary_mult * risk_mult
        
        # Determine if career meets core constraints
        meets_constraints = salary_ok and risk_ok
        
        scored_careers.append({
            'career': career,
            'nlp_score': nlp_score,
            'final_score': final_score,
            'salary_fit': salary_ok,
            'risk_fit': risk_ok,
            'meets_all': meets_constraints
        })
    
    # Sort by final score (NLP + constraints combined)
    scored_careers.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Build output
    recommended = []
    horizon_key = TIME_HORIZON_MAP.get(time_horizon, "entry")
    
    for item in scored_careers[:3]:  # Top 3
        career = item['career']
        salary = career['salaries'].get(horizon_key, 0)
        
        # Build reason from NLP match + career description
        reason_parts = [career['description'].split('.')[0] + '.']  # First sentence
        reason_parts.append(f"Projected {horizon_key}-level salary: {salary} LPA.")
        reason_parts.append(f"Match confidence: {item['final_score']:.0%}.")
        
        recommended.append({
            "role": career['role'],
            "reason": ' '.join(reason_parts)
        })
    
    # Generate feasibility note
    feasibility_note = generate_feasibility_note(
        scored_careers, salary_range, time_horizon, risk_appetite
    )
    
    return recommended, feasibility_note


def generate_feasibility_note(scored_careers: list, salary_range: str,
                              time_horizon: str, risk_appetite: str) -> str:
    """Generate helpful feasibility note based on results."""
    
    # Count how many meet all constraints
    meets_all = sum(1 for c in scored_careers if c['meets_all'])
    
    if meets_all >= 3:
        return "Recommendations matched using TF-IDF similarity with high confidence."
    
    if meets_all == 0:
        # No perfect matches - explain why
        notes = []
        
        if salary_range == "premium" and time_horizon == "immediate":
            notes.append(
                "Premium salaries are rarely achievable immediately. "
                "Consider extending timeline to mid_term or long_term."
            )
        
        if salary_range == "premium" and risk_appetite == "low":
            notes.append(
                "Low-risk premium careers are rare. "
                "Consider PSU Officer or Chartered Accountant for stable high pay."
            )
        
        if not notes:
            notes.append(
                "Limited matches for your exact criteria. "
                "Showing closest matches based on NLP similarity."
            )
        
        return ' '.join(notes)
    
    return f"Found {meets_all} strong matches. Showing top results ranked by NLP similarity."
