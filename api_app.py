from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.classifier_service import (
    predict_product_identity as classify_product
)
from services.openai_service import (
    generate_product_concepts as generate_products
)

from services.evaluation_service import (
    evaluate_product as evaluate_product_with_llm
)

# ============================================================
# Create FastAPI application
# ============================================================

app = FastAPI(
    title="AI-Powered Gen Z Credit Card Product Innovation API",
    description=(
        "Backend API for generating, validating, and evaluating "
        "AI-powered Canadian credit card concepts for Gen Z personas."
    ),
    version="2.0"
)


# ============================================================
# Product Specification Input
# Must match the exact classifier feature set
# ============================================================

class ProductSpecification(BaseModel):

    annual_fee_cad: float
    minimum_income_requirement: float
    minimum_credit_score: float
    foreign_transaction_fee_percent: float
    purchase_interest_rate: float
    base_reward_rate: float
    welcome_bonus_value_cad: float

    travel_feature_level: float
    cashback_feature_level: float
    digital_feature_level: float
    subscription_feature_level: float
    lifestyle_feature_level: float
    sustainability_feature_level: float
    financial_wellness_feature_level: float
    security_protection_feature_level: float
    premium_benefits_feature_level: float

# ============================================================
# Product Generation Request
# ============================================================

class ProductGenerationRequest(BaseModel):

    target_persona: str

    number_of_concepts: int = Field(
        default=3,
        ge=1,
        le=5
    )

# ============================================================
# Product Evaluation Request Models
# ============================================================

class GeneratedProductEvidence(BaseModel):

    product_name: str
    llm_proposed_category: str
    concept_theme: str

    annual_fee_cad: float
    minimum_income_requirement: float
    minimum_credit_score: float
    purchase_interest_rate: float
    foreign_transaction_fee_percent: float
    base_reward_rate: float

    reward_program_type: str
    reward_architecture: str
    customer_value_proposition: str
    original_business_justification: str


class EngineeredFeatureLevels(BaseModel):

    travel_feature_level: float
    cashback_feature_level: float
    digital_feature_level: float
    subscription_feature_level: float
    lifestyle_feature_level: float
    sustainability_feature_level: float
    financial_wellness_feature_level: float
    security_protection_feature_level: float
    premium_benefits_feature_level: float


class SimilarityEvidence(BaseModel):

    closest_existing_card: str
    closest_existing_category: str
    closest_existing_cluster: int | str
    highest_similarity: float
    similarity_interpretation: str


class ClassificationEvidence(BaseModel):

    predicted_product_identity: str


class ProductEvaluationRequest(BaseModel):

    target_persona: str
    generated_product: GeneratedProductEvidence
    engineered_feature_levels: EngineeredFeatureLevels
    similarity_analysis: SimilarityEvidence
    product_identity_classification: ClassificationEvidence
    
# ============================================================
# Home Endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": (
            "AI-Powered Gen Z Credit Card Product "
            "Innovation System is running successfully!"
        ),
        "version": "2.0"
    }


# ============================================================
# Product Identity Classification
# ============================================================

@app.post("/predict-product-identity")
def predict_product_identity(
    product: ProductSpecification
):

    try:
        product_data = product.model_dump()

        predicted_category = classify_product(
            product_data
        )

        return {
            "predicted_product_identity": predicted_category
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Product identity prediction failed: "
                f"{str(error)}"
            )
        )


# ============================================================
# Generate AI Product Concepts
# ============================================================

@app.post("/ai-product-generation")
def generate_product_concepts(
    request: ProductGenerationRequest
):

    try:
        result = generate_products(
            request.model_dump()
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI product generation failed: "
                f"{str(error)}"
            )
        )
# ============================================================
# AI Product Evaluation
# ============================================================

@app.post("/evaluate-product")
def evaluate_product(
    request: ProductEvaluationRequest
):

    try:
        evidence = request.model_dump()

        evaluation_result = evaluate_product_with_llm(
            evidence
        )

        return evaluation_result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI product evaluation failed: "
                f"{str(error)}"
            )
        )
