import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from alerts import check_early_warning

st.set_page_config(
    page_title="Heart Disease Risk App",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_models():
    logreg = joblib.load("models/logreg_baseline.joblib")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgb_model.json")
    return logreg, xgb_model

@st.cache_data
def load_median_values():
    df = pd.read_csv("data/heart.csv")
    medians = {
        "age": int(df["age"].median()),
        "trestbps": int(df["trestbps"].median()),
        "chol": int(df["chol"].median()),
        "thalach": int(df["thalach"].median()),
        "oldpeak": float(df["oldpeak"].median()),
        "ca": int(df["ca"].median()),
    }
    return medians

def encode_input(
    age: int, sex: str, cp: str, trestbps: int, chol: int,
    fbs: str, restecg: str, thalach: int, exang: str,
    oldpeak: float, slope: str, ca: int, thal: str
) -> pd.DataFrame:
    # Numeric features
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    # Map chest pain type → numeric code 0–3
    cp_code = {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal pain": 2,
        "asymptomatic": 3
    }[cp]
    # Create 3 dummies: cp_1, cp_2, cp_3
    if cp_code == 0:
        cp_vals = [0, 0, 0]
    elif cp_code == 1:
        cp_vals = [1, 0, 0]
    elif cp_code == 2:
        cp_vals = [0, 1, 0]
    else:  # cp_code == 3
        cp_vals = [0, 0, 1]

    # Map resting ECG → numeric code 0–2
    restecg_code = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }[restecg]
    # Create 2 dummies: restecg_1, restecg_2
    if restecg_code == 0:
        restecg_vals = [0, 0]
    elif restecg_code == 1:
        restecg_vals = [1, 0]
    else:  # restecg_code == 2
        restecg_vals = [0, 1]

    # Map slope → numeric code 0–2
    slope_code = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }[slope]
    # Create 2 dummies: slope_1, slope_2
    if slope_code == 0:
        slope_vals = [0, 0]
    elif slope_code == 1:
        slope_vals = [1, 0]
    else:  # slope_code == 2
        slope_vals = [0, 1]

    # Map thalassemia → numeric code 0–3 (Kaggle: 0=unknown, 1=fixed defect, 2=reversible, 3=normal)
    thal_code = {
        "Fixed defect": 1,
        "Reversible defect": 2,
        "Normal": 3
    }[thal]
    # Create 3 dummies: thal_1, thal_2, thal_3
    if thal_code == 0:
        thal_vals = [0, 0, 0]
    elif thal_code == 1:
        thal_vals = [1, 0, 0]
    elif thal_code == 2:
        thal_vals = [0, 1, 0]
    else:  # thal_code == 3
        thal_vals = [0, 0, 1]

    # Assemble feature vector in exact order:
    #
    # ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    #  'cp_1', 'cp_2', 'cp_3',
    #  'restecg_1', 'restecg_2',
    #  'slope_1', 'slope_2',
    #  'thal_1', 'thal_2', 'thal_3']
    feature_vector = [
        age,
        sex_val,
        trestbps,
        chol,
        fbs_val,
        thalach,
        exang_val,
        oldpeak,
        ca,
        *cp_vals,
        *restecg_vals,
        *slope_vals,
        *thal_vals
    ]

    _, xgb_model = load_models()
    feature_names = xgb_model.get_booster().feature_names
    X_new = pd.DataFrame([feature_vector], columns=feature_names)
    return X_new


def main():
    st.title("❤️ Heart Disease Risk App")
    st.markdown(
        """
        Adjust the sliders in the sidebar to see how each factor affects your
        predicted risk of heart disease. Any critical “red-flag” values like
        extremely high blood pressure or cholesterol will trigger an urgent alert.
        """
    )


    logreg_model, xgb_model = load_models()

    with st.sidebar:
        st.header("Patient Profile")

        medians = load_median_values()

        if "reset" not in st.session_state:
            st.session_state["reset"] = False

        if st.button("🔄 Reset to Median Values"):
            st.session_state["reset"] = True

        age_default = medians["age"] if st.session_state["reset"] else 50
        trestbps_default = medians["trestbps"] if st.session_state["reset"] else 120
        chol_default = medians["chol"] if st.session_state["reset"] else 200
        thalach_default = medians["thalach"] if st.session_state["reset"] else 150
        oldpeak_default = medians["oldpeak"] if st.session_state["reset"] else 1.0
        ca_default = medians["ca"] if st.session_state["reset"] else 0

        age = st.slider(
            "Age", min_value=29, max_value=77, value=age_default, help="Age in years"
        )
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest pain type (cp)",
            ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
        )
        trestbps = st.slider(
            "Resting BP (mm Hg)", min_value=94, max_value=200, value=trestbps_default, help="Resting blood pressure"
        )
        chol = st.slider(
            "Cholesterol (mg/dL)", min_value=126, max_value=564, value=chol_default, help="Serum cholesterol"
        )
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dL?", ["Yes", "No"])
        restecg = st.selectbox(
            "Resting ECG results",
            ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
        )
        thalach = st.slider(
            "Max heart rate achieved", min_value=71, max_value=202, value=thalach_default
        )
        exang = st.selectbox("Exercise-induced angina?", ["Yes", "No"])
        oldpeak = st.slider(
            "ST depression (oldpeak)", min_value=0.0, max_value=6.2, value=oldpeak_default, step=0.1
        )
        slope = st.selectbox(
            "Slope of peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"]
        )
        ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3], index=ca_default)
        thal = st.selectbox("Thalassemia", ["Fixed defect", "Reversible defect", "Normal"])

        if st.session_state["reset"]:
            st.session_state["reset"] = False

        compute_button = st.button("Compute Risk")

    if compute_button:
        raw_feats = {
            "age": age,
            "trestbps": trestbps,
            "chol": chol
        }

        warnings = check_early_warning(raw_feats)
        if warnings:
            for msg in warnings:
                st.error(msg)
        else:
            st.success("No immediate red-flag values detected.")

        X_new = encode_input(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal
        )

        prob = xgb_model.predict_proba(X_new)[0, 1]
        st.subheader(f"Predicted Risk of Heart Disease: {prob:.1%}")

        fig, ax = plt.subplots(figsize=(6, 0.6))
        bar_color = "crimson" if prob >= 0.5 else "seagreen"
        ax.barh([0], [prob], color=bar_color)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Probability")
        ax.set_title("Risk Gauge (≥ 50% considered high risk)")
        st.pyplot(fig)

        coef_age = logreg_model.coef_[0][0]
        intercept = logreg_model.intercept_[0]
        p = np.clip(prob, 1e-6, 1 - 1e-6)
        raw_heart_age = (np.log(p / (1 - p)) - intercept) / coef_age

        # Clamp negative values to zero
        heart_age = max(raw_heart_age, 0)

        st.markdown(f"**Chronological age**: {age}  \n**Heart age**: {heart_age:.1f}")

        # st.markdown(
        #     """
        #     _“Heart age” is the age at which an average person would have the same baseline‐only risk.  
        #     For example, if your predicted risk is 20% (0.20), we find the age where the baseline model 
        #     (using only age) also gives 20%._
        #     """
        # )

if __name__ == "__main__":
    main()
