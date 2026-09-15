# ============================================================
# OpenAI Product Generation Service
# ============================================================

import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not available in the environment."
    )

client = OpenAI(
    api_key=api_key
)


# ============================================================
# Data-Driven Gen Z Persona Profiles
# ============================================================

persona_profiles = {

    "Affluent Lifestyle Explorers": {
        "cluster": 0,
        "profile": {
            "average_age": 23.48,
            "average_annual_income": 49600.69,
            "average_credit_score": 727.36,
            "average_credit_limit": 5079.15,
            "average_utilization_rate": 0.10,
            "pays_in_full_probability": 0.69,
            "everyday_spending_score": 751.30,
            "lifestyle_spending_score": 1515.88,
            "digital_banking_score": 0.80,
            "digital_engagement_score": 0.87,
            "financial_wellness_score": 0.70,
            "reward_preference_score": 0.65,
            "experience_lifestyle_score": 0.52,
            "sustainability_score": 0.63,
            "credit_spend_share": 0.57,
            "lifestyle_spend_share": 0.67
        },
        "characteristics": [
            "Highest income among the four Gen Z segments",
            "High credit score and strong credit capacity",
            "Highest everyday and lifestyle spending",
            "High lifestyle spending share",
            "Strong digital engagement",
            "Generally able to pay balances in full",
            "Interested in experiences, convenience, and lifestyle value"
        ],
        "product_opportunity": (
            "Explore premium lifestyle, experience, flexible rewards, "
            "travel, dining, entertainment, or differentiated concepts."
        )
    },

    "Digital Creators": {
        "cluster": 1,
        "profile": {
            "average_age": 21.71,
            "average_annual_income": 23806.51,
            "average_credit_score": 706.05,
            "average_credit_limit": 3276.39,
            "average_utilization_rate": 0.06,
            "pays_in_full_probability": 0.72,
            "everyday_spending_score": 304.24,
            "lifestyle_spending_score": 581.11,
            "social_media_engagement_score": 1.34,
            "social_commerce_score": 0.66,
            "digital_banking_score": 0.81,
            "digital_engagement_score": 0.93,
            "financial_wellness_score": 0.72,
            "creator_economy_score": 0.58,
            "credit_spend_share": 0.59,
            "lifestyle_spend_share": 0.65
        },
        "characteristics": [
            "Lower average income but strong financial behaviour",
            "Highest digital engagement",
            "Highest digital banking score",
            "Highest social-commerce activity",
            "Highest social-media engagement",
            "Highest creator-economy participation",
            "High probability of paying balances in full",
            "Values seamless digital experiences and online commerce"
        ],
        "product_opportunity": (
            "Explore creator-economy, digital-first, social-commerce, "
            "subscription, online-shopping, or platform-partnership concepts."
        )
    },

    "Financially Responsible Young Professionals": {
        "cluster": 2,
        "profile": {
            "average_age": 23.74,
            "average_annual_income": 36060.26,
            "average_credit_score": 745.69,
            "average_credit_limit": 4735.19,
            "average_utilization_rate": 0.05,
            "pays_in_full_probability": 0.72,
            "everyday_spending_score": 475.03,
            "lifestyle_spending_score": 821.51,
            "digital_banking_score": 0.80,
            "digital_engagement_score": 0.83,
            "financial_wellness_score": 0.72,
            "financial_pressure_score": 0.43,
            "reward_preference_score": 0.65,
            "sustainability_score": 0.60,
            "credit_spend_share": 0.60,
            "lifestyle_spend_share": 0.63
        },
        "characteristics": [
            "Highest average credit score",
            "Highest average credit limit",
            "Strong financial wellness",
            "Low credit-utilization rate",
            "High probability of paying balances in full",
            "Moderate-to-high income",
            "Interested in rewards while maintaining responsible habits",
            "Potential interest in wealth building and long-term value"
        ],
        "product_opportunity": (
            "Explore cashback, flexible rewards, financial wellness, "
            "wealth-building, savings-linked, or professional-lifestyle concepts."
        )
    },

    "Budget-Conscious Students": {
        "cluster": 3,
        "profile": {
            "average_age": 20.89,
            "average_annual_income": 21575.35,
            "average_credit_score": 677.21,
            "average_credit_limit": 2414.10,
            "average_utilization_rate": 0.09,
            "pays_in_full_probability": 0.65,
            "everyday_spending_score": 281.72,
            "lifestyle_spending_score": 525.98,
            "digital_banking_score": 0.79,
            "digital_engagement_score": 0.84,
            "financial_wellness_score": 0.68,
            "financial_pressure_score": 0.46,
            "reward_preference_score": 0.66,
            "sustainability_score": 0.64,
            "credit_spend_share": 0.57,
            "lifestyle_spend_share": 0.65
        },
        "characteristics": [
            "Youngest Gen Z segment",
            "Lowest average income",
            "Lowest average credit score and credit limit",
            "Higher financial pressure",
            "Strong interest in rewards",
            "High digital engagement despite limited financial capacity",
            "Needs accessible credit-building and money-management support",
            "Likely sensitive to fees and eligibility requirements"
        ],
        "product_opportunity": (
            "Explore accessible student, credit-building, budgeting, "
            "everyday-value, subscription, or financial-wellness concepts."
        )
    }
}


# ============================================================
# AI Context Builder
# ============================================================

def build_ai_context(request: dict) -> dict:

    target_persona = request["target_persona"]

    number_of_concepts = request.get(
        "number_of_concepts",
        3
    )

    if target_persona not in persona_profiles:
        raise ValueError(
            "Unknown target persona. Valid personas are: "
            + ", ".join(persona_profiles.keys())
        )

    persona = persona_profiles[target_persona]

    return {
        "target_persona": target_persona,
        "source_customer_cluster": persona["cluster"],
        "persona_profile": persona["profile"],
        "persona_characteristics": persona["characteristics"],
        "product_opportunity": persona["product_opportunity"],
        "number_of_concepts": number_of_concepts
    }


# ============================================================
# Prompt Builder
# ============================================================

def build_product_generation_prompt(
    ai_context: dict
) -> str:

    characteristics_text = "\n".join(
        f"- {item}"
        for item in ai_context[
            "persona_characteristics"
        ]
    )

    profile_text = json.dumps(
        ai_context["persona_profile"],
        indent=2
    )

    prompt = f"""
You are a senior Canadian credit card product innovation specialist.

OBJECTIVE

Generate {ai_context["number_of_concepts"]} distinct and innovative
consumer credit card concepts for the selected Gen Z persona.

TARGET PERSONA

{ai_context["target_persona"]}

SOURCE CUSTOMER CLUSTER

Cluster {ai_context["source_customer_cluster"]}

PERSONA CHARACTERISTICS

{characteristics_text}

CUSTOMER CLUSTER PROFILE

{profile_text}

GENERAL PRODUCT OPPORTUNITY

{ai_context["product_opportunity"]}

REQUIREMENTS

Independently determine:

- Product name
- Product category
- Concept theme
- Annual fee
- Minimum income requirement
- Minimum credit score
- Purchase interest rate
- Foreign transaction fee
- Base reward rate
- Welcome bonus
- Reward program type
- Reward architecture
- Digital features
- Reward features
- Subscription features
- Lifestyle features
- Sustainability features
- Financial-wellness features
- Security and protection features
- Premium benefits
- Customer value proposition
- Business justification

Generate meaningfully different concepts.
Do not create minor variations of the same card.
Do not copy existing card names.
Do not evaluate similarity or predict product identity.
Do not claim proven profitability, adoption, or customer demand.

Return only valid JSON using this structure:

{{
  "target_persona": "{ai_context["target_persona"]}",
  "concepts": [
    {{
      "product_name": "string",
      "product_category": "string",
      "concept_theme": "string",
      "annual_fee_cad": 0.0,
      "minimum_income_requirement": 0,
      "minimum_credit_score": 0,
      "purchase_interest_rate": 0.0,
      "foreign_transaction_fee_percent": 0.0,
      "base_reward_rate": 0.0,
      "welcome_bonus_value_cad": 0.0,
      "reward_program_type": "string",
      "reward_architecture": "string",

      "travel_rewards": 0,
      "airport_lounge_access": 0,
      "travel_insurance": 0,
      "trip_cancellation_insurance": 0,
      "baggage_insurance": 0,
      "rental_car_insurance": 0,
      "no_foreign_transaction_fee": 0,
      "hotel_rewards": 0,
      "airline_rewards": 0,

      "cashback_rewards": 0,
      "grocery_rewards": 0,
      "dining_rewards": 0,
      "fuel_rewards": 0,
      "transit_rewards": 0,
      "shopping_rewards": 0,

      "mobile_wallet_support": 0,
      "apple_pay_support": 0,
      "google_pay_support": 0,
      "samsung_pay_support": 0,
      "virtual_card_support": 0,
      "instant_card_issuance": 0,
      "digital_first_card": 0,

      "subscription_rewards": 0,
      "streaming_rewards": 0,
      "netflix_benefit": 0,
      "spotify_benefit": 0,
      "disney_plus_benefit": 0,
      "amazon_prime_benefit": 0,
      "youtube_premium_benefit": 0,
      "food_delivery_benefit": 0,
      "uber_benefit": 0,
      "instacart_benefit": 0,
      "doordash_benefit": 0,
      "skip_benefit": 0,

      "ride_share_rewards": 0,
      "gaming_rewards": 0,
      "entertainment_rewards": 0,

      "sustainability_rewards": 0,
      "carbon_tracking": 0,
      "green_rewards_program": 0,
      "sustainable_purchase_rewards": 0,

      "credit_score_monitoring": 0,
      "budgeting_tools": 0,
      "spending_insights": 0,
      "installment_payment_option": 0,
      "debt_management_support": 0,

      "fraud_alerts": 0,
      "identity_protection": 0,
      "purchase_protection": 0,
      "zero_liability": 0,

      "concierge_service": 0,
      "premium_hotel_benefits": 0,
      "luxury_travel_credits": 0,
      "exclusive_event_access": 0,

      "customer_value_proposition": "string",
      "business_justification": "string"
    }}
  ]
}}

For every binary feature:

- Use 1 when the feature is included.
- Use 0 when the feature is not included.
- Use integers only.
- Include every field.
"""

    return prompt.strip()


# ============================================================
# Parse OpenAI JSON
# ============================================================

def parse_json_response(
    response_text: str
) -> dict:

    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]

    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    return json.loads(
        cleaned_text.strip()
    )


# ============================================================
# Generate Product Concepts
# ============================================================

def generate_product_concepts(
    request: dict
) -> dict:

    ai_context = build_ai_context(
        request
    )

    generation_prompt = (
        build_product_generation_prompt(
            ai_context
        )
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=generation_prompt
    )

    result = parse_json_response(
        response.output_text
    )

    return result