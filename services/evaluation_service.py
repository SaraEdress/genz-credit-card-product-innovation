# ============================================================
# AI Product Evaluation Service
# ============================================================

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Load OpenAI API key
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not available in the environment."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# Build Evaluation Prompt
# ============================================================

def build_evaluation_prompt(evidence: dict) -> str:

    evidence_text = json.dumps(
        evidence,
        indent=2,
        default=str
    )

    prompt = f"""
You are a senior Canadian banking product evaluator.

Your role is to evaluate an AI-generated consumer credit card concept
using independent analytical evidence produced by the product evaluation
pipeline.

IMPORTANT LIMITATIONS

- Do not redesign the product.
- Do not invent new product features.
- Do not recalculate engineered feature levels.
- Do not change the similarity score.
- Do not override the product identity classifier.
- Do not claim that the product is profitable.
- Do not claim proven customer demand or adoption.
- Use only the evidence supplied below.
- Evaluate the product only against its selected persona, stated value
  proposition, proposed category, pricing, and analytical evidence.
- Do not treat the absence of unrelated feature groups as a weakness.
- Do not criticize the absence of travel, premium, sustainability, or
  lifestyle features unless they are relevant to the persona or concept.
- A narrowly targeted product may still be a strong recommendation.

EVIDENCE

{evidence_text}

EVALUATION TASK

1. Persona Fit
Assess whether the concept appears suitable for the selected Gen Z persona.

2. Feature Evidence
Explain whether the engineered feature levels support the proposed value
proposition and concept theme.

3. Classification Consistency
Compare the LLM-proposed product category with the predicted product identity.
State whether they are consistent, partially consistent, or inconsistent.

4. Market Differentiation
Interpret the closest existing product and similarity score.

5. Strengths
Identify the strongest evidence-supported aspects.

6. Weaknesses and Risks
Identify only evidence-supported concerns, such as:
- High similarity
- Weak differentiation
- Persona misalignment
- Classification inconsistency
- Accessibility concerns
- Pricing or eligibility concerns
- Unsupported business claims
- Features that do not support the stated proposition

7. Final Recommendation
Choose:
- Proceed
- Revise
- Do Not Proceed

OUTPUT REQUIREMENTS

Return only valid JSON using exactly this structure:

{{
  "product_name": "string",
  "persona_fit_rating": "Strong",
  "persona_fit_assessment": "string",
  "feature_evidence_assessment": "string",
  "classification_consistency_rating": "Consistent",
  "classification_consistency_assessment": "string",
  "market_differentiation_rating": "Moderate",
  "market_differentiation_assessment": "string",
  "strengths": [
    "string",
    "string"
  ],
  "weaknesses_or_risks": [
    "string",
    "string"
  ],
  "final_recommendation": "Revise",
  "recommendation_rationale": "string"
}}

ALLOWED VALUES

persona_fit_rating:
- Strong
- Moderate
- Weak

classification_consistency_rating:
- Consistent
- Partially Consistent
- Inconsistent

market_differentiation_rating:
- High
- Moderate
- Low

final_recommendation:
- Proceed
- Revise
- Do Not Proceed
"""

    return prompt.strip()


# ============================================================
# Parse OpenAI JSON
# ============================================================

def parse_json_response(response_text: str) -> dict:

    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]

    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    return json.loads(cleaned_text.strip())


# ============================================================
# Evaluate Product
# ============================================================

def evaluate_product(
    evidence: dict,
    model: str = "gpt-4.1-mini"
) -> dict:

    prompt = build_evaluation_prompt(evidence)

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return parse_json_response(
        response.output_text
    )