import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🧠",
    layout="centered",
)
st.title("🧠 Student Mental Health — Depression Prediction")
st.write(
    "Enter the student's information below. The trained **CatBoost** model "
    "will estimate the likelihood of depression."
)

# ---------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model("model.joblib")
    st.success("✅ model.joblib loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Introspect the model: get exact feature order & categories
#    CatBoost stores cat-feature category vocab in `get_cat_feature_indices`
#    and we can read the fit-time feature names if it was a DataFrame.
# ---------------------------------------------------------
FEATURES = [
    "Gender",
    "Age",
    "City",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Sleep Duration int",
]

CATEGORICAL = [
    "Gender",
    "City",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Financial Stress",
    "Family History of Mental Illness",
]

# Try to read training vocabulary (only works if model was fit on a DataFrame
# or if you saved the encoder separately — optional, safe to fail).
def try_extract_vocab(model):
    vocab = {}
    try:
        if hasattr(model, "feature_names_"):
            pass  # list of names only
        # CatBoost stores cat features by name; sometimes stores category list
        if hasattr(model, "get_cat_feature_indices"):
            _ = model.get_cat_feature_indices()  # triggers lazy init; ignore result
    except Exception:
        pass
    return vocab

TRAIN_VOCAB = try_extract_vocab(model)

# Fallbacks derived from the notebook (head + describe)
GENDER_OPTS = TRAIN_VOCAB.get("Gender", ["Male", "Female"])
CITY_OPTS = TRAIN_VOCAB.get("City", [
    "Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata",
    "Ludhiana", "Mumbai", "Pune", "Srinagar", "Varanasi", "Visakhapatnam",
])
DEGREE_OPTS = TRAIN_VOCAB.get("Degree", [
    "BA", "B.Pharm", "B.Sc", "BCA", "BE", "BSc", "B.Tech",
    "M.Tech", "MBA", "ME", "M.Sc",
])
DIET_OPTS = TRAIN_VOCAB.get("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
SUICIDAL_OPTS = TRAIN_VOCAB.get("Have you ever had suicidal thoughts ?", ["No", "Yes"])
FAMILY_OPTS = TRAIN_VOCAB.get("Family History of Mental Illness", ["No", "Yes"])
# Financial Stress was a STRING in training ("1.0".."5.0")
FIN_STRESS_OPTS = TRAIN_VOCAB.get("Financial Stress", ["1.0", "2.0", "3.0", "4.0", "5.0"])
SLEEP_OPTS = [5, 6, 7, 8]  # derived from `Sleep Duration int` after to_numeric

# Debug expander — shows you exactly what the dropdowns are offering
with st.expander("🔎 Training vocabulary (debug)", expanded=False):
    st.write("**Gender**", GENDER_OPTS)
    st.write("**City**", CITY_OPTS)
    st.write("**Degree**", DEGREE_OPTS)
    st.write("**Dietary Habits**", DIET_OPTS)
    st.write("**Financial Stress**", FIN_STRESS_OPTS)
    st.caption(
        "If CatBoost throws 'unknown categories', replace the lists above "
        "with the values your notebook actually trained on."
    )

# ---------------------------------------------------------
# 4. Input widgets
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", GENDER_OPTS)
    age = st.number_input("Age", min_value=18, max_value=59, value=25, step=1)
    city = st.selectbox("City", CITY_OPTS)
    academic_pressure = st.slider("Academic Pressure (0–5)", 0.0, 5.0, 3.0, 0.5)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
    study_satisfaction = st.slider("Study Satisfaction (0–5)", 0.0, 5.0, 3.0, 0.5)
    dietary_habits = st.selectbox("Dietary Habits", DIET_OPTS)

with col2:
    degree = st.selectbox("Degree", DEGREE_OPTS)
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", SUICIDAL_OPTS)
    work_study_hours = st.number_input("Work/Study Hours per day",
                                       min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    # Financial Stress: keep as STRING to match training dtype
    financial_stress = st.selectbox("Financial Stress", FIN_STRESS_OPTS)
    family_history = st.selectbox("Family History of Mental Illness", FAMILY_OPTS)
    sleep_duration_int = st.selectbox("Sleep Duration (hours, averaged)", SLEEP_OPTS)

# ---------------------------------------------------------
# 5. Build input_df with EXACT names, order and dtypes
#    - numeric  -> float
#    - categorical -> str  (CatBoost will encode internally)
# ---------------------------------------------------------
input_df = pd.DataFrame([{
    "Gender":                               str(gender),
    "Age":                                  float(age),
    "City":                                 str(city),
    "Academic Pressure":                    float(academic_pressure),
    "CGPA":                                 float(cgpa),
    "Study Satisfaction":                   float(study_satisfaction),
    "Dietary Habits":                       str(dietary_habits),
    "Degree":                               str(degree),
    "Have you ever had suicidal thoughts ?": str(suicidal_thoughts),
    "Work/Study Hours":                     float(work_study_hours),
    "Financial Stress":                     str(financial_stress),   # ⚠️ STRING
    "Family History of Mental Illness":     str(family_history),
    "Sleep Duration int":                   float(sleep_duration_int),
}], columns=FEATURES)   # force the exact training column order

with st.expander("🔎 Model input"):
    st.dataframe(input_df, use_container_width=True)
    st.caption(f"dtypes → {dict(input_df.dtypes.astype(str))}")

# ---------------------------------------------------------
# 6. Predict
# ---------------------------------------------------------
if st.button("🔮 Predict", type="primary", use_container_width=True):
    try:
        prediction = int(model.predict(input_df)[0])
        label_map = {0: "Not Depressed 🟢", 1: "Depressed 🔴"}
        pretty = label_map.get(prediction, str(prediction))

        st.subheader("Result")
        st.success(f"### Prediction: **{pretty}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.info(f"Confidence: **{np.max(proba) * 100:.2f}%**")
            st.bar_chart(
                pd.DataFrame(
                    {"Probability": proba},
                    index=[label_map.get(i, str(i)) for i in range(len(proba))],
                )
            )
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)
