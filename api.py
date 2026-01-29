"""
Career Recommendation REST API.

Flask server exposing the NLP-based recommendation engine.
Designed for cloud deployment on Railway, Render, or similar platforms.

Endpoints:
    POST /recommend - Get career recommendations
    GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from logic import get_recommendations, validate_inputs, ValidationError

app = Flask(__name__)

# Enable CORS for all origins (required for cross-origin frontend requests)
# This allows the frontend hosted on GitHub Pages to call this API
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Career recommendation endpoint using TF-IDF NLP.
    
    Request Body (JSON):
        {
            "salary_range": "entry | growth | premium",
            "time_horizon": "immediate | mid_term | long_term",
            "risk_appetite": "low | medium | high",
            "skills": "optional free-text skills" (optional)
        }
    
    Success Response (200):
        {
            "recommended_careers": [
                {"role": "...", "reason": "..."}
            ],
            "feasibility_note": "..."
        }
    
    Error Response (400):
        {"error": "...", "field": "..."}
    """
    try:
        # Parse JSON body
        data = request.get_json()
        
        if data is None:
            return jsonify({
                "error": "Invalid or missing JSON body"
            }), 400
        
        # Validate inputs
        validated = validate_inputs(data)
        
        # Get NLP-based recommendations
        recommendations, feasibility_note = get_recommendations(
            validated["salary_range"],
            validated["time_horizon"],
            validated["risk_appetite"],
            validated.get("skills")
        )
        
        # Build response
        response = {
            "recommended_careers": recommendations,
            "feasibility_note": feasibility_note
        }
        
        return jsonify(response), 200
        
    except ValidationError as e:
        return jsonify({
            "error": e.message,
            "field": e.field
        }), 400
        
    except Exception as e:
        # Log error for debugging
        print(f"[ERROR] {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Used by frontend to verify backend connectivity.
    """
    return jsonify({"status": "ok"}), 200


@app.route('/debug/terms', methods=['POST'])
def debug_terms():
    """
    Debug endpoint to see TF-IDF query terms.
    Useful for understanding how NLP matching works.
    """
    try:
        from nlp_model import get_nlp_model
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400
        
        validated = validate_inputs(data)
        
        nlp_model = get_nlp_model()
        query_info = nlp_model.get_query_terms(
            validated["salary_range"],
            validated["time_horizon"],
            validated["risk_appetite"],
            validated.get("skills")
        )
        
        return jsonify({
            "query_text": query_info["query_text"],
            "top_terms": [{"term": t, "weight": round(w, 4)} 
                         for t, w in query_info["top_terms"]]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os
if __name__ == '__main__':
    print("=" * 60)
    print("Career Recommendation API Server (TF-IDF NLP)")
    print("=" * 60)
    print("Endpoints:")
    print("  POST /recommend  - Get career recommendations")
    print("  GET  /health     - Health check")
    print("  POST /debug/terms - View TF-IDF query terms (debug)")
    print("=" * 60)
    print()
    
    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

