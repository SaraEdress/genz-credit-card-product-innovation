import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="AI Product Innovation",
    page_icon="💳",
    layout="wide"
)
# ============================================================
# Custom Banking Dashboard Theme
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #F5F7FA;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: #16324F;
        font-family: Arial, sans-serif;
    }

    p, li, label {
        color: #263746;
    }

    /* Banking-style header */
    .bank-header {
        background-color: #16324F;
        border-bottom: 5px solid #F4A261;
        padding: 22px 30px;
        border-radius: 4px 4px 0 0;
        margin-bottom: 0;
    }

    .bank-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }

    .bank-header p {
        color: #D9E7F0;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Navigation bar */
    .nav-container {
        background-color: white;
        border-bottom: 1px solid #D5DEE6;
        padding: 8px 18px;
        margin-bottom: 28px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    /* Standard buttons */
    div.stButton > button {
        background-color: white;
        color: #16324F;
        border: 1px solid #B8C6D1;
        border-radius: 4px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background-color: #EAF3F8;
        color: #16324F;
        border-color: #247BA0;
    }

    /* Primary action buttons */
    div.stButton > button[kind="primary"] {
        background-color: #247BA0;
        color: white;
        border: 1px solid #247BA0;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1B6584;
        color: white;
        border-color: #1B6584;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #D9E1E8;
        border-top: 4px solid #247BA0;
        border-radius: 6px;
        padding: 18px;
        box-shadow: 0 2px 7px rgba(0, 0, 0, 0.05);
    }

    /* Information cards */
    .info-card {
        background-color: white;
        border: 1px solid #D9E1E8;
        border-radius: 6px;
        padding: 22px;
        margin-bottom: 18px;
        min-height: 180px;
        box-shadow: 0 2px 7px rgba(0, 0, 0, 0.04);
    }

    .info-card h3 {
        color: #16324F;
        margin-top: 0;
    }

    .feature-label {
        display: inline-block;
        background-color: #EAF3F8;
        color: #16324F;
        padding: 5px 10px;
        border-radius: 14px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Section heading */
    .section-title {
        border-left: 5px solid #F4A261;
        padding-left: 12px;
        margin-top: 20px;
        margin-bottom: 18px;
    }

    /* Footer */
    .project-footer {
        margin-top: 42px;
        padding: 24px;
        background-color: #16324F;
        color: white;
        text-align: center;
        border-top: 5px solid #F4A261;
    }

    .project-footer strong {
        color: white;
    }

    .project-footer p {
        color: #D9E7F0;
        margin: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
PROJECT_ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PRODUCT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_canadian_consumer_credit_card_product_intelligence_v3.csv"
)

# ============================================================
# Project Information
# ============================================================

PROJECT_TITLE = (
    "AI-Powered Gen Z Credit Card "
    "Product Innovation System"
)

PROGRAM_NAME = (
    "Artificial Intelligence – "
    "Integration and Governance"
)

INSTITUTION_NAME = "Humber Polytechnic"

PROJECT_TEAM = [
    "Sara Edress",
    "Onyema Oshinowo"
]

PROJECT_SPONSOR = "Effie Sismanis"

PYTHON_VERSION = "3.11+"

PRODUCT_DATABASE_RECORDS = 1357

PRODUCT_FEATURE_LEVELS = 9

# ============================================================
# Gen Z Persona Profiles
# (K-Means, k=4, from Notebook 2 — Gen Z Customer Segmentation)
# Values are actual cluster means from the trained model, used to
# demonstrate that the four personas are behaviourally distinct.
# ============================================================

PERSONA_PROFILES = {
    "Affluent Lifestyle Explorers": {
        "avg_age": 23.5,
        "avg_income": 49601,
        "avg_credit_score": 727,
        "avg_credit_limit": 5079,
        "pays_in_full_rate": 0.69,
        "defining_trait": "Lifestyle spending is 180% of the Gen Z average",
        "product_opportunity": "Premium lifestyle / travel rewards"
    },
    "Digital Creators": {
        "avg_age": 21.7,
        "avg_income": 23807,
        "avg_credit_score": 706,
        "avg_credit_limit": 3276,
        "pays_in_full_rate": 0.72,
        "defining_trait": "Highest social-commerce and creator-economy engagement (112% of average)",
        "product_opportunity": "Creator-economy / digital-first card"
    },
    "Financially Responsible Young Professionals": {
        "avg_age": 23.7,
        "avg_income": 36060,
        "avg_credit_score": 746,
        "avg_credit_limit": 4735,
        "pays_in_full_rate": 0.72,
        "defining_trait": "Highest credit score and strongest financial wellness in the population",
        "product_opportunity": "Premium cashback / wealth-building card"
    },
    "Budget-Conscious Students": {
        "avg_age": 20.9,
        "avg_income": 21575,
        "avg_credit_score": 677,
        "avg_credit_limit": 2414,
        "pays_in_full_rate": 0.65,
        "defining_trait": "Highest credit utilization and financial pressure in the population",
        "product_opportunity": "Low-fee, student credit-building card"
    }
}

# ============================================================
# Random Forest Classifier — Top Feature Importances
# (from Notebook 3 — Product Identity Classification;
# RandomForestClassifier, n_estimators=300, test accuracy 94.5%)
# ============================================================

CLASSIFIER_ACCURACY = 0.945

CLASSIFIER_FEATURE_IMPORTANCE = {
    "Purchase interest rate": 0.167,
    "Cashback feature level": 0.138,
    "Subscription feature level": 0.137,
    "Sustainability feature level": 0.073,
    "Welcome bonus value": 0.065,
    "Base reward rate": 0.063,
    "Annual fee": 0.061
}

# ============================================================
# Session State
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "generated_products" not in st.session_state:
    st.session_state.generated_products = None
    
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if "analytical_evidence" not in st.session_state:
    st.session_state.analytical_evidence = None

# ============================================================
# Product Feature Groups
# ============================================================

FEATURE_GROUPS = {

    "travel_feature_level": [
        "travel_rewards",
        "airport_lounge_access",
        "travel_insurance",
        "trip_cancellation_insurance",
        "baggage_insurance",
        "rental_car_insurance",
        "no_foreign_transaction_fee",
        "hotel_rewards",
        "airline_rewards"
    ],

    "cashback_feature_level": [
        "cashback_rewards",
        "grocery_rewards",
        "dining_rewards",
        "fuel_rewards",
        "transit_rewards",
        "shopping_rewards"
    ],

    "digital_feature_level": [
        "mobile_wallet_support",
        "apple_pay_support",
        "google_pay_support",
        "samsung_pay_support",
        "virtual_card_support",
        "instant_card_issuance",
        "digital_first_card"
    ],

    "subscription_feature_level": [
        "subscription_rewards",
        "streaming_rewards",
        "netflix_benefit",
        "spotify_benefit",
        "disney_plus_benefit",
        "amazon_prime_benefit",
        "youtube_premium_benefit",
        "food_delivery_benefit",
        "uber_benefit",
        "instacart_benefit",
        "doordash_benefit",
        "skip_benefit"
    ],

    "lifestyle_feature_level": [
        "grocery_rewards",
        "dining_rewards",
        "fuel_rewards",
        "transit_rewards",
        "ride_share_rewards",
        "gaming_rewards",
        "entertainment_rewards",
        "shopping_rewards",
        "food_delivery_benefit"
    ],

    "sustainability_feature_level": [
        "sustainability_rewards",
        "carbon_tracking",
        "green_rewards_program",
        "sustainable_purchase_rewards"
    ],

    "financial_wellness_feature_level": [
        "credit_score_monitoring",
        "budgeting_tools",
        "spending_insights",
        "installment_payment_option",
        "debt_management_support"
    ],

    "security_protection_feature_level": [
        "fraud_alerts",
        "identity_protection",
        "purchase_protection",
        "zero_liability"
    ],

    "premium_benefits_feature_level": [
        "airport_lounge_access",
        "concierge_service",
        "premium_hotel_benefits",
        "luxury_travel_credits",
        "exclusive_event_access"
    ]
}

CLASSIFICATION_FEATURES = [
    "welcome_bonus_value_cad",
    "annual_fee_cad",
    "minimum_income_requirement",
    "minimum_credit_score",
    "foreign_transaction_fee_percent",
    "purchase_interest_rate",
    "base_reward_rate",
    "cashback_feature_level",
    "lifestyle_feature_level",
    "travel_feature_level",
    "digital_feature_level",
    "subscription_feature_level",
    "sustainability_feature_level",
    "financial_wellness_feature_level",
    "security_protection_feature_level",
    "premium_benefits_feature_level"
]

SIMILARITY_FEATURES = CLASSIFICATION_FEATURES.copy()

# ============================================================
# Feature Engineering
# ============================================================

def calculate_feature_levels(product: dict) -> dict:

    engineered_product = product.copy()

    for output_column, source_columns in FEATURE_GROUPS.items():

        missing_columns = [
            column
            for column in source_columns
            if column not in engineered_product
        ]

        if missing_columns:
            raise ValueError(
                f"Missing source features for {output_column}: "
                f"{missing_columns}"
            )

        values = [
            float(engineered_product[column])
            for column in source_columns
        ]

        engineered_product[output_column] = round(
            np.mean(values) * 100,
            2
        )

    return engineered_product

 
    
# ============================================================
# Database Preparation
# ============================================================

@st.cache_data
def load_product_database():

    database = pd.read_csv(
        PRODUCT_DATABASE_FILE
    )

    if "card_category" in database.columns:

        database = database[
            database["card_category"]
            .astype(str)
            .str.lower()
            .ne("business")
        ].copy()

    for output_column, source_columns in FEATURE_GROUPS.items():

        if output_column not in database.columns:

            missing_columns = [
                column
                for column in source_columns
                if column not in database.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Cannot calculate {output_column}. "
                    f"Database is missing: {missing_columns}"
                )

            database[output_column] = (
                database[source_columns]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
                .mul(100)
                .round(2)
            )

    return database
    
# ============================================================
# Similarity Analysis
# ============================================================

def find_first_column(dataframe, possible_names):

    for column in possible_names:
        if column in dataframe.columns:
            return column

    return None


def run_similarity_analysis(
    engineered_product: dict,
    product_database: pd.DataFrame
) -> dict:

    missing_database_features = [
        column
        for column in SIMILARITY_FEATURES
        if column not in product_database.columns
    ]

    if missing_database_features:
        raise ValueError(
            "Product database is missing similarity features: "
            f"{missing_database_features}"
        )

    generated_df = pd.DataFrame([
        engineered_product
    ])

    database_numeric = product_database[
        SIMILARITY_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    generated_numeric = generated_df[
        SIMILARITY_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    medians = database_numeric.median()

    database_numeric = database_numeric.fillna(
        medians
    )

    generated_numeric = generated_numeric.fillna(
        medians
    )

    database_mean = database_numeric.mean()

    database_std = (
        database_numeric
        .std()
        .replace(0, 1)
    )

    database_scaled = (
        database_numeric - database_mean
    ) / database_std

    generated_scaled = (
        generated_numeric - database_mean
    ) / database_std

    similarities = cosine_similarity(
        generated_scaled,
        database_scaled
    )[0]

    closest_position = int(
        np.argmax(similarities)
    )

    closest_product = product_database.iloc[
        closest_position
    ]

    similarity_score = float(
        similarities[closest_position]
    )

    product_name_column = find_first_column(
        product_database,
        [
            "card_name",
            "product_name",
            "card_product_name"
        ]
    )

    category_column = find_first_column(
        product_database,
        [
            "card_category",
            "product_category"
        ]
    )

    cluster_column = find_first_column(
        product_database,
        [
            "product_cluster",
            "cluster"
        ]
    )

    if similarity_score >= 0.90:
        interpretation = (
            "Very high similarity: the concept strongly "
            "resembles an existing product."
        )

    elif similarity_score >= 0.75:
        interpretation = (
            "High similarity: the concept shares many "
            "characteristics with an existing product."
        )

    elif similarity_score >= 0.55:
        interpretation = (
            "Moderate similarity: the concept combines "
            "familiar and differentiated characteristics."
        )

    else:
        interpretation = (
            "Low similarity: the concept appears relatively "
            "differentiated in the product database."
        )

    return {
        "closest_existing_card": (
            str(closest_product[product_name_column])
            if product_name_column
            else f"Product {closest_position}"
        ),
        "closest_existing_category": (
            str(closest_product[category_column])
            if category_column
            else "Unknown"
        ),
        "closest_existing_cluster": (
            str(closest_product[cluster_column])
            if cluster_column
            else "Unknown"
        ),
        "highest_similarity": round(
            similarity_score,
            4
        ),
        "similarity_interpretation": interpretation
    }    
# ============================================================
# LLM Evaluation API
# ============================================================

def request_llm_evaluation(
    target_persona: str,
    engineered_product: dict,
    similarity_result: dict,
    predicted_identity: str
) -> dict:

    evidence_payload = {

        "target_persona": target_persona,

        "generated_product": {
            "product_name": engineered_product[
                "product_name"
            ],
            "llm_proposed_category": engineered_product[
                "product_category"
            ],
            "concept_theme": engineered_product[
                "concept_theme"
            ],
            "annual_fee_cad": engineered_product[
                "annual_fee_cad"
            ],
            "minimum_income_requirement": engineered_product[
                "minimum_income_requirement"
            ],
            "minimum_credit_score": engineered_product[
                "minimum_credit_score"
            ],
            "purchase_interest_rate": engineered_product[
                "purchase_interest_rate"
            ],
            "foreign_transaction_fee_percent": engineered_product[
                "foreign_transaction_fee_percent"
            ],
            "base_reward_rate": engineered_product[
                "base_reward_rate"
            ],
            "reward_program_type": engineered_product[
                "reward_program_type"
            ],
            "reward_architecture": engineered_product[
                "reward_architecture"
            ],
            "customer_value_proposition": engineered_product[
                "customer_value_proposition"
            ],
            "original_business_justification": engineered_product[
                "business_justification"
            ]
        },

        "engineered_feature_levels": {
            column: engineered_product[column]
            for column in FEATURE_GROUPS.keys()
        },

        "similarity_analysis": similarity_result,

        "product_identity_classification": {
            "predicted_product_identity": predicted_identity
        }
    }

    response = requests.post(
        f"{API_URL}/evaluate-product",
        json=evidence_payload,
        timeout=120
    )

    response.raise_for_status()

    return {
        "evaluation": response.json(),
        "evidence": evidence_payload
    }
    
# ============================================================
# Product Identity Classification API
# ============================================================

def predict_product_identity(
    engineered_product: dict
) -> str:

    classification_payload = {
        column: engineered_product[column]
        for column in CLASSIFICATION_FEATURES
    }

    response = requests.post(
        f"{API_URL}/predict-product-identity",
        json=classification_payload,
        timeout=60
    )

    response.raise_for_status()

    return response.json()[
        "predicted_product_identity"
    ]
# ============================================================
# Project Header
# ============================================================

def display_project_header():

    st.markdown(
        f"""
        <div class="bank-header">
            <h1>{PROJECT_TITLE}</h1>
            <p>
                Responsible AI and machine learning decision support
                for Canadian banking product teams
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# Top Navigation
# ============================================================

def display_navigation():

    st.markdown(
        '<div class="nav-container">',
        unsafe_allow_html=True
    )

    nav_1, nav_2, nav_3, nav_4 = st.columns(4)

    with nav_1:

        if st.button(
            "Home",
            use_container_width=True,
            key="nav_home"
        ):
            st.session_state.page = "home"
            st.rerun()

    with nav_2:

        if st.button(
            "Product Innovation",
            use_container_width=True,
            key="nav_innovation"
        ):
            st.session_state.page = "persona"
            st.rerun()

    with nav_3:

        if st.button(
            "About the Project",
            use_container_width=True,
            key="nav_project"
        ):
            st.session_state.page = "about_project"
            st.rerun()

    with nav_4:

        if st.button(
            "About Us",
            use_container_width=True,
            key="nav_us"
        ):
            st.session_state.page = "about_us"
            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
# ============================================================
# Project Footer
# ============================================================

def display_project_footer():

    team_names = " and ".join(
        PROJECT_TEAM
    )

    st.markdown(
        f"""
        <div class="project-footer">
            <p><strong>{PROJECT_TITLE}</strong></p>
            <p>{PROGRAM_NAME} · {INSTITUTION_NAME}</p>
            <p>
                Project Team: {team_names}
                &nbsp; | &nbsp;
                Sponsor: {PROJECT_SPONSOR}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# Home Page
# ============================================================
display_project_header()
display_navigation()
if st.session_state.page == "home":

    st.title("💳 AI-Powered Gen Z Credit Card Product Innovation System")

    st.markdown("""
Develop innovative Canadian credit card concepts using Artificial Intelligence and Machine Learning.

This system supports banking product teams by:

- Selecting a Gen Z customer persona
- Generating innovative credit card concepts
- Validating products using AI and Machine Learning
- Producing evidence-based product recommendations
""")

    st.caption(
        "Built on synthetic Canadian data as a proof of concept for "
        "AI-assisted decision support — not a production approval system."
    )

    st.divider()

    if st.button("Start Product Innovation", use_container_width=True):

        st.session_state.page = "persona"
        st.rerun()

# ============================================================
# Persona Selection
# ============================================================

elif st.session_state.page == "persona":

    st.title("Select Target Gen Z Persona")

    st.write(
        "Four personas were identified from K-Means clustering on the "
        "synthetic Gen Z customer dataset. Their financial and behavioural "
        "profiles are measurably different, which is why each one calls "
        "for a different card design."
    )

    with st.expander("See how these four personas differ", expanded=False):

        profile_table = pd.DataFrame(PERSONA_PROFILES).T

        st.dataframe(
            profile_table[
                [
                    "avg_age",
                    "avg_income",
                    "avg_credit_score",
                    "avg_credit_limit",
                    "pays_in_full_rate",
                    "defining_trait"
                ]
            ].rename(columns={
                "avg_age": "Avg Age",
                "avg_income": "Avg Income (CAD)",
                "avg_credit_score": "Avg Credit Score",
                "avg_credit_limit": "Avg Credit Limit (CAD)",
                "pays_in_full_rate": "Pays-in-Full Rate",
                "defining_trait": "Defining Trait"
            }),
            use_container_width=True
        )

        st.bar_chart(
            profile_table["avg_income"].rename("Average Annual Income (CAD)")
        )

        st.caption(
            "Annual income alone ranges from roughly $21.6k to $49.6k across "
            "personas — a 2.3x spread — with similarly wide gaps in credit "
            "score, spending, and digital behaviour. Source: K-Means "
            "segmentation (k=4), Notebook 2."
        )

    persona = st.radio(

        "Choose a customer segment:",

        (
            "Affluent Lifestyle Explorers",
            "Digital Creators",
            "Financially Responsible Young Professionals",
            "Budget-Conscious Students"
        )

    )

    if st.button("Generate AI Products", use_container_width=True):

        with st.spinner("Generating innovative products..."):

            response = requests.post(

                f"{API_URL}/ai-product-generation",

                json={
                    "target_persona": persona,
                    "number_of_concepts": 3
                }

            )

            if response.status_code == 200:

                st.session_state.generated_products = response.json()

                st.session_state.page = "products"

                st.rerun()

            else:

                st.error(response.text)


    
# ============================================================
# Generated Products Page
# ============================================================

elif st.session_state.page == "products":

    st.title("AI-Generated Credit Card Concepts")

    generated_data = st.session_state.generated_products

    if not generated_data:
        st.error("No generated products are available.")

        if st.button("Return to Persona Selection"):
            st.session_state.page = "persona"
            st.rerun()

    else:
        target_persona = generated_data.get(
            "target_persona",
            "Selected Persona"
        )

        concepts = generated_data.get(
            "concepts",
            []
        )

        st.write(
            f"Target Persona: **{target_persona}**"
        )

        st.write(
            f"Generated Concepts: **{len(concepts)}**"
        )

        st.divider()

        if not concepts:
            st.warning(
                "The API returned no product concepts."
            )

        else:
            columns = st.columns(
                len(concepts)
            )

            for index, product in enumerate(concepts):

                with columns[index]:

                    st.subheader(
                        product.get(
                            "product_name",
                            f"Concept {index + 1}"
                        )
                    )

                    st.caption(
                        product.get(
                            "product_category",
                            "Category not provided"
                        )
                    )

                    st.metric(
                        "Annual Fee",
                        f"${product.get('annual_fee_cad', 0):,.2f}"
                    )

                    reward_type = str(
                        product.get("reward_program_type", "")
                    ).lower()
                    
                    base_rate = float(
                        product.get("base_reward_rate", 0)
                    )
                    
                    if "cashback" in reward_type:
                    
                        reward_display = (
                            f"{base_rate:.2f}% Cashback"
                        )
                    
                    elif "point" in reward_type:
                    
                        reward_display = (
                            f"{base_rate:.2f} Points per $1"
                        )
                    
                    elif "mile" in reward_type:
                    
                        reward_display = (
                            f"{base_rate:.2f} Miles per $1"
                        )
                    
                    else:
                    
                        reward_display = (
                            f"{base_rate:.2f} Rewards per $1"
                        )
                    
                    st.metric(
                        "Base Rewards",
                        reward_display
                    )
                    
                    st.caption(
                        f"Reward Program: {product.get('reward_program_type', 'N/A')}"
                    )
                    st.write(
                        "**Concept Theme**"
                    )

                    st.write(
                        product.get(
                            "concept_theme",
                            "Not provided"
                        )
                    )

                    st.write(
                        "**Reward Architecture**"
                    )

                    st.write(
                        product.get(
                            "reward_architecture",
                            "Not provided"
                        )
                    )

                    st.write(
                        "**Customer Value Proposition**"
                    )

                    st.write(
                        product.get(
                            "customer_value_proposition",
                            "Not provided"
                        )
                    )

                    if st.button(
                        "Select for Evaluation",
                        key=f"evaluate_{index}",
                        use_container_width=True
                    ):
                    
                        st.session_state.selected_product = product
                    
                        st.session_state.evaluation_result = None
                        st.session_state.analytical_evidence = None
                    
                        st.session_state.page = "evaluation"
                    
                        st.rerun()

        st.divider()

        if st.button(
            "Generate Different Concepts"
        ):
            st.session_state.generated_products = None
            st.session_state.page = "persona"
            st.rerun()

# ============================================================
# Product Evaluation Page
# ============================================================

elif st.session_state.page == "evaluation":

    selected_product = st.session_state.selected_product
    generated_data = st.session_state.generated_products

    if selected_product is None or generated_data is None:

        st.error(
            "No product has been selected for evaluation."
        )

        if st.button("Return to Product Selection"):
            st.session_state.page = "products"
            st.rerun()

        st.stop()

    target_persona = generated_data.get(
        "target_persona",
        "Selected Persona"
    )

    st.title("AI Product Evaluation")

    st.subheader(
        selected_product.get(
            "product_name",
            "Selected Product"
        )
    )

    st.write(
        f"Target Persona: **{target_persona}**"
    )

    st.write(
        "This evaluation uses feature engineering, similarity analysis, "
        "product identity classification, and LLM evidence interpretation."
    )

    st.divider()

    # ========================================================
    # Run Evaluation
    # ========================================================

    if st.session_state.evaluation_result is None:

        if st.button(
            "Run Product Evaluation",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Engineering features, running similarity analysis, "
                    "predicting product identity, and generating evaluation..."
                ):

                    # 1. Feature engineering
                    engineered_product = calculate_feature_levels(
                        selected_product
                    )

                    # 2. Load product intelligence database
                    product_database = load_product_database()

                    # 3. Similarity analysis
                    similarity_result = run_similarity_analysis(
                        engineered_product,
                        product_database
                    )

                    # 4. Product identity classification
                    predicted_identity = predict_product_identity(
                        engineered_product
                    )

                    # 5. LLM evidence-based evaluation
                    evaluation_package = request_llm_evaluation(
                        target_persona=target_persona,
                        engineered_product=engineered_product,
                        similarity_result=similarity_result,
                        predicted_identity=predicted_identity
                    )

                    st.session_state.evaluation_result = (
                        evaluation_package["evaluation"]
                    )

                    st.session_state.analytical_evidence = (
                        evaluation_package["evidence"]
                    )

                st.rerun()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Confirm that Uvicorn is running on port 8000."
                )

            except requests.exceptions.HTTPError as error:

                if error.response is not None:
                    st.error(
                        f"API error: {error.response.text}"
                    )
                else:
                    st.error(
                        f"API error: {str(error)}"
                    )

            except FileNotFoundError:

                st.error(
                    f"Product database file not found: "
                    f"{PRODUCT_DATABASE_FILE}"
                )

            except Exception as error:

                st.error(
                    f"Evaluation failed: {str(error)}"
                )

    # ========================================================
    # Display Basic Evaluation Results
    # ========================================================

    else:

        evaluation = st.session_state.evaluation_result
        evidence = st.session_state.analytical_evidence

        similarity = evidence["similarity_analysis"]

        classification = evidence[
            "product_identity_classification"
        ]

        st.success(
            "Product evaluation completed successfully."
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:

            st.metric(
                "Similarity",
                f"{similarity['highest_similarity'] * 100:.2f}%"
            )

        with metric_2:

            st.metric(
                "Persona Fit",
                evaluation["persona_fit_rating"]
            )

        with metric_3:

            st.metric(
                "Recommendation",
                evaluation["final_recommendation"]
            )

        st.divider()

        st.subheader("Product Validation")

        left_column, right_column = st.columns(2)

        with left_column:

            st.write("**Predicted Product Identity**")

            st.write(
                classification[
                    "predicted_product_identity"
                ]
            )

            st.write("**Classification Consistency**")

            st.write(
                evaluation[
                    "classification_consistency_rating"
                ]
            )

        with right_column:

            st.write("**Closest Existing Product**")

            st.write(
                similarity[
                    "closest_existing_card"
                ]
            )

            st.write("**Market Differentiation**")

            st.write(
                evaluation[
                    "market_differentiation_rating"
                ]
            )

        with st.expander("Why this classification? (Random Forest evidence)"):

            st.write(
                f"The product identity classifier is a Random Forest "
                f"(300 trees) with **{CLASSIFIER_ACCURACY * 100:.1f}% "
                f"accuracy** on held-out test data. It never sees this "
                f"concept's category label — it infers it purely from "
                f"the engineered pricing and feature-level values below, "
                f"ranked by how much each one drives its decisions."
            )

            st.bar_chart(
                pd.Series(CLASSIFIER_FEATURE_IMPORTANCE).sort_values()
            )

            st.caption(
                "Feature importances from the trained classifier "
                "(Notebook 3), shown as-is — not recalculated per request."
            )

        st.divider()

        st.subheader("Recommendation Rationale")

        st.write(
            evaluation[
                "recommendation_rationale"
            ]
        )

        st.divider()

        if st.button(
            "Back to Generated Products",
            use_container_width=True
        ):

            st.session_state.evaluation_result = None
            st.session_state.analytical_evidence = None
            st.session_state.page = "products"
            st.rerun()
# ============================================================
# About the Project Page
# ============================================================

elif st.session_state.page == "about_project":

    st.markdown(
        """
        <div class="section-title">
            <h2>About the Project</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        This capstone project develops an AI-powered decision-support
        system for Canadian banking product teams. The system uses
        Gen Z customer intelligence, machine learning, generative AI,
        and product validation models to support the development of
        new consumer credit card concepts.
        """
    )

    st.divider()

    overview_1, overview_2, overview_3 = st.columns(3)

    with overview_1:

        st.metric(
            "Product Intelligence Records",
            f"{PRODUCT_DATABASE_RECORDS:,}"
        )

    with overview_2:

        st.metric(
            "Engineered Feature Levels",
            PRODUCT_FEATURE_LEVELS
        )

    with overview_3:

        st.metric(
            "Python Version",
            PYTHON_VERSION
        )

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            <h2>System Architecture</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    architecture_columns = st.columns(4)

    architecture_items = [
        (
            "Customer Intelligence",
            "K-Means customer segmentation identifies "
            "four evidence-based Gen Z personas."
        ),
        (
            "AI Product Generation",
            "OpenAI generates structured Canadian "
            "credit card concepts for the selected persona."
        ),
        (
            "Independent Validation",
            "Feature engineering, cosine similarity, and "
            "Random Forest classification validate each concept."
        ),
        (
            "Evaluation Engine",
            "An LLM interprets analytical evidence and returns "
            "Proceed, Revise, or Do Not Proceed."
        )
    ]

    for column, item in zip(
        architecture_columns,
        architecture_items
    ):

        title, description = item

        with column:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="section-title">
            <h2>Product Intelligence Database</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        The Product Intelligence Database contains synthetic Canadian
        consumer credit card products. Business credit cards were
        removed so the analysis focuses on consumer products relevant
        to Gen Z customers.
        """
    )

    st.markdown(
        """
        <span class="feature-label">Pricing</span>
        <span class="feature-label">Eligibility</span>
        <span class="feature-label">Rewards</span>
        <span class="feature-label">Digital Features</span>
        <span class="feature-label">Subscriptions</span>
        <span class="feature-label">Lifestyle</span>
        <span class="feature-label">Sustainability</span>
        <span class="feature-label">Financial Wellness</span>
        <span class="feature-label">Security</span>
        <span class="feature-label">Premium Benefits</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
            <h2>Machine Learning and AI Models</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_data = {
        "Component": [
            "Customer Segmentation",
            "Product Clustering",
            "Product Identity Classification",
            "Similarity Engine",
            "AI Product Generation",
            "AI Product Evaluation"
        ],
        "Method": [
            "K-Means Clustering",
            "K-Means Clustering",
            "Random Forest Classifier",
            "Standardization + Cosine Similarity",
            "OpenAI LLM",
            "Evidence-Based LLM Evaluation"
        ],
        "Purpose": [
            "Identify Gen Z personas",
            "Identify product identities",
            "Predict generated-product identity",
            "Find the closest existing product",
            "Generate new credit card concepts",
            "Interpret evidence and recommend next action"
        ],
        "Measured Performance": [
            "4 segments, silhouette 0.069",
            "6 product clusters",
            "94.5% test accuracy",
            "Cosine similarity, 0-1 scale",
            "gpt-4.1-mini",
            "gpt-4.1-mini, evidence-constrained"
        ]
    }

    st.dataframe(
        pd.DataFrame(model_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="section-title">
            <h2>Responsible AI Design Choices</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "This system is built on **synthetic data** and is a proof of "
        "concept for AI-assisted decision support — not a production "
        "approval system. Two design choices keep it honest and "
        "auditable:"
    )

    responsible_1, responsible_2 = st.columns(2)

    with responsible_1:

        st.markdown(
            """
            <div class="info-card">
                <h3>A defensible segmentation trade-off</h3>
                <p>
                    The statistically optimal split of the Gen Z dataset
                    was 2 clusters (silhouette score 0.114). We chose
                    4 clusters instead (silhouette score 0.069) because
                    four personas are far more actionable for a bank than
                    two broad groups — a deliberate, disclosed trade-off
                    between statistical fit and business usefulness.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with responsible_2:

        st.markdown(
            """
            <div class="info-card">
                <h3>An LLM that can't override the evidence</h3>
                <p>
                    The evaluation prompt explicitly instructs the LLM
                    not to redesign the product, invent new features,
                    recalculate the engineered feature levels, change
                    the similarity score, or override the classifier.
                    It may only interpret evidence that the ML models
                    already produced — keeping the final recommendation
                    traceable back to real analysis, not free-form
                    generation.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Every recommendation in this tool is decision support for a "
        "human product team, not an automated approval."
    )

    st.markdown(
        """
        <div class="section-title">
            <h2>Technology Stack</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <span class="feature-label">Python {PYTHON_VERSION}</span>
        <span class="feature-label">Pandas</span>
        <span class="feature-label">NumPy</span>
        <span class="feature-label">Scikit-learn</span>
        <span class="feature-label">FastAPI</span>
        <span class="feature-label">Streamlit</span>
        <span class="feature-label">OpenAI API</span>
        <span class="feature-label">Joblib</span>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# About Us Page
# ============================================================

elif st.session_state.page == "about_us":

    st.markdown(
        """
        <div class="section-title">
            <h2>About Us</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        This system was developed as a capstone project within
        Humber Polytechnic's Artificial Intelligence – Integration
        and Governance program.
        """
    )

    team_column, sponsor_column = st.columns(2)

    with team_column:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>Project Team</h3>
                <p><strong>{PROJECT_TEAM[0]}</strong></p>
                <p><strong>{PROJECT_TEAM[1]}</strong></p>
                <p>
                    Responsible for customer intelligence,
                    product intelligence, machine learning,
                    generative AI integration, product validation,
                    and application development.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with sponsor_column:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>Project Sponsor</h3>
                <p><strong>{PROJECT_SPONSOR}</strong></p>
                <p>
                    Provided project direction, banking product
                    context, review feedback, and guidance regarding
                    the practical needs of product teams.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            <h2>Academic Program</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
            <h3>{INSTITUTION_NAME}</h3>
            <p><strong>{PROGRAM_NAME}</strong></p>
            <p>
                The project demonstrates applied machine learning,
                responsible generative AI integration, explainability,
                governance-aware product design, and full-stack AI
                application development.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

display_project_footer()
