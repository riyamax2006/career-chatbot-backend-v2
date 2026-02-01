"""
Business Logic for Career Recommendations.

This module handles:
1. Input validation
2. Domain purity filtering
3. NLP gating (0.30 threshold)
4. Strict scoring (70% NLP, 15% Salary, 15% Risk)
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
    """Validate and normalize all input parameters."""
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    
    salary_range = data.get("salary_range")
    if salary_range is None:
        raise ValidationError("salary_range is required", field="salary_range")
    salary_range = str(salary_range).lower().strip()
    if salary_range not in VALID_SALARY_RANGES:
        raise ValidationError(f"Invalid salary_range. Must be: {VALID_SALARY_RANGES}", field="salary_range")
    
    time_horizon = data.get("time_horizon")
    if time_horizon is None:
        raise ValidationError("time_horizon is required", field="time_horizon")
    time_horizon = str(time_horizon).lower().strip()
    if time_horizon not in VALID_TIME_HORIZONS:
        raise ValidationError(f"Invalid time_horizon. Must be: {VALID_TIME_HORIZONS}", field="time_horizon")
    
    risk_appetite = data.get("risk_appetite")
    if risk_appetite is None:
        raise ValidationError("risk_appetite is required", field="risk_appetite")
    risk_appetite = str(risk_appetite).lower().strip()
    if risk_appetite not in VALID_RISK_LEVELS:
        raise ValidationError(f"Invalid risk_appetite. Must be: {VALID_RISK_LEVELS}", field="risk_appetite")
    
    skills = data.get("skills", "")
    if skills:
        skills = str(skills).strip()
    
    return {
        "salary_range": salary_range,
        "time_horizon": time_horizon,
        "risk_appetite": risk_appetite,
        "skills": skills
    }


def calculate_salary_score(career: dict, salary_range: str, time_horizon: str) -> float:
    """Calculate normalized salary score (0-1)."""
    horizon_key = TIME_HORIZON_MAP.get(time_horizon, "entry")
    salary = career["salaries"].get(horizon_key, 0)
    min_sal, max_sal = SALARY_BOUNDS.get(salary_range, (0, float("inf")))
    
    # Simple linear penalties, but never override NLP
    if min_sal <= salary <= max_sal:
        return 1.0 # Perfect
    elif max_sal != float("inf") and salary > max_sal:
        return 0.8 # earning more is good, but maybe mismatch
    
    # Earning less than target
    if min_sal > 0:
        ratio = salary / min_sal
        return max(0.2, ratio) # scale down
        
    return 0.5


def calculate_risk_score(career: dict, risk_appetite: str) -> float:
    """Calculate normalized risk score (0-1)."""
    career_risk = career["risk"].lower()
    risk_levels = ["low", "medium", "high"]
    
    try:
        c_idx = risk_levels.index(career_risk)
        u_idx = risk_levels.index(risk_appetite)
        
        # User is ok with this risk or higher
        if c_idx <= u_idx:
            return 1.0
        
        # Risk is higher than user wants - penalty
        diff = c_idx - u_idx
        return max(0.2, 1.0 - (diff * 0.3))
        
    except ValueError:
        return 0.5


def get_domain_constraints(skills: str) -> list:
    """
    Check if skills contain explicit domain keywords.
    Returns list of allowed career tags/categories if found.
    """
    skills_lower = skills.lower()
    
    # Define domain mappings
    domains = {
        "medical": ["medical", "healthcare", "doctor", "health"],
        "healthcare": ["medical", "healthcare", "doctor", "health"],
        "hospital": ["medical", "healthcare", "doctor", "health"],
        "tech": ["technology", "engineering", "software", "it", "computer", "data"],
        "programming": ["technology", "engineering", "software", "it"],
        "finance": ["finance", "banking", "investment", "accounting"],
        "business": ["business", "management", "consulting"],
        "law": ["legal", "law"],
    }
    
    constraints = set()
    for key, keywords in domains.items():
        if key in skills_lower:
             constraints.add(key)
    
    return list(constraints)


def check_domain_match(career: dict, constraints: list) -> bool:
    """Return True if career matches ANY of the constraints."""
    if not constraints:
        return True
        
    # Check text fields for domain presence
    text = (career['role'] + " " + career['category'] + " " + career.get('description', '')).lower()
    
    for domain in constraints:
        # Check against simple heuristic of domain mapping
        # This is a simplified check. Ideally we use the category.
        if domain == "medical" or domain == "healthcare":
            if "health" in text or "doctor" in text or "medic" in text or "nurse" in text:
                return True
        elif domain == "tech" or domain == "programming":
            if "tech" in text or "software" in text or "developer" in text or "engineer" in text or "data" in text:
                return True
        elif domain == "finance":
            if "finance" in text or "bank" in text or "invest" in text:
                return True
        # Add other domains as needed
        
        # Fallback: check if domain keyword itself is in text
        if domain in text:
            return True
            
    return False


def get_recommendations(salary_range: str, time_horizon: str, 
                        risk_appetite: str, skills: str = None) -> tuple:
    """Score and rank careers enforcing NLP dominance."""
    
    # 1. Get raw NLP scores
    nlp_model = get_nlp_model()
    # Pass ONLY skills to NLP model (logic logic handles inputs)
    nlp_results = nlp_model.get_recommendations(skills, top_k=None)
    
    # 2. Domain Purity Filter
    domain_constraints = get_domain_constraints(skills) if skills else []
    
    filtered_step1 = []
    for res in nlp_results:
        if check_domain_match(res['career'], domain_constraints):
            filtered_step1.append(res)
            
    # If strictly filtering removed everything (rare but possible),
    # we might need to relax, but prompt says "ONLY careers from that domain".
    # So we strictly return empty or next best if nothing matches?
    # Requirement: "If no careers pass this threshold, return top 3 NLP matches".
    # But Domain Purity says "ONLY careers from that domain are allowed".
    # Domain purity > Threshold fallback.
    
    if not filtered_step1 and domain_constraints:
        # If user asked for "healthcare" and we have none, we return none.
        # But we assume the dataset has some.
        pass

    # 3. NLP Threshold Gating
    candidates = []
    for res in filtered_step1:
        if res['similarity_score'] >= 0.30:
            candidates.append(res)
            
    # Fallback: If no careers pass threshold, return top 3 NLP matches anyway
    # (But they must still pass domain purity if it exists)
    if not candidates:
        # Take from filtered_step1 (which are domain-pure)
        candidates = filtered_step1[:3]
        
    # 4. Scoring
    scored_results = []
    W_NLP = 0.70
    W_SALARY = 0.15
    W_RISK = 0.15
    
    for item in candidates:
        career = item['career']
        nlp_score = item['similarity_score']
        
        sal_score = calculate_salary_score(career, salary_range, time_horizon)
        risk_score = calculate_risk_score(career, risk_appetite)
        
        total_score = (W_NLP * nlp_score) + (W_SALARY * sal_score) + (W_RISK * risk_score)
        
        scored_results.append({
            "career": career,
            "total_score": total_score,
            "nlp_score": nlp_score,
            "sal_score": sal_score,
            "risk_score": risk_score
        })
        
    # Sort
    scored_results.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Top 3
    final_results = scored_results[:3]
    
    # 5. Output Formatting
    recommended = []
    horizon_key = TIME_HORIZON_MAP.get(time_horizon, "entry")
    
    for item in final_results:
        c = item['career']
        salary = c['salaries'].get(horizon_key, 0)
        
        reason = (f"Score: {item['total_score']:.2f} "
                  f"(NLP: {item['nlp_score']:.2f}, "
                  f"Sal: {item['sal_score']:.2f}, "
                  f"Risk: {item['risk_score']:.2f})")
        
        recommended.append({
            "role": c['role'],
            "reason": reason
        })
        
    # Feasibility Note
    if not recommended:
        note = "No matching careers found for this domain/query."
    else:
        top = final_results[0]
        if top['nlp_score'] < 0.30:
            note = "Low relevance matches. Try broader terms."
        else:
            note = "Matches found based on your skills and preferences."

    return recommended, note
