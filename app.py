# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Thyroid Cancer Lymph Node Metastasis Prediction System",
    page_icon="🏥",
    layout="wide"
)

# --- Custom CSS (保持不变) ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #2e86ab; margin-top: 1.5rem; margin-bottom: 1rem; }
    .prediction-box { background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 10px 0; }
    .high-risk { background-color: #ffebee; border-left: 5px solid #f44336; }
    .low-risk { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
</style>
""", unsafe_allow_html=True)


class ThyroidCancerPredictor:
    def __init__(self):
        self.model = None
        self.model_info = None
        self.load_model()

    def load_model(self):
        try:
            with open('models/catboost_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            st.sidebar.success("✅ Model Loaded")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.info("Please run `train_model.py` first.")

    def predict(self, input_data):
        if self.model is None:
            return None, None
        try:
            input_df = pd.DataFrame([input_data])[self.model_info['features']]
            probability = self.model.predict_proba(input_df)[0, 1]
            prediction = 1 if probability > 0.5 else 0
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None


def create_input_form():
    st.markdown('<div class="sub-header">📋 Patient Clinical Features Input</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Ultrasound & Location")
        New_focal = st.selectbox(
            "New Lesion Type",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Unilateral single-focus", 2: "Unilateral multifocal", 3: "Bilateral lesions"}[x]
        )
        CDFI = st.selectbox(
            "Color Doppler Flow Signal",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        Boundary = st.selectbox(
            "Boundary Clarity",
            options=[0, 1],
            format_func=lambda x: "Clear" if x == 0 else "Unclear"
        )
        Special_location = st.selectbox(
            "Special Tumor Location",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Pole & Isthmus", 1: "Near Trachea", 2: "Middle"}[x]
        )

    with col2:
        st.markdown("#### Pathological Features")
        NG = st.selectbox(
            "Nodular Goiter",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        HT = st.selectbox(
            "Hashimoto's Thyroiditis",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        Microcalcification = st.selectbox(
            "Microcalcification",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        Tumor_size_custom = st.selectbox(
            "Tumor Size",
            options=[1, 2, 3],
            format_func=lambda x: {1: "<10mm", 2: "10-20mm", 3: "≥20mm"}[x]
        )

    with col3:
        st.markdown("#### Laboratory Indicators")
        TSH = st.number_input(
            "TSH Level (mU/L)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )
        SIRI_four = st.selectbox(
            "SIRI Score Grade",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "<P25", 2: "P25-P50", 3: "P50-P75", 4: "≥P75"}[x]
        )
        LMR_four = st.selectbox(
            "LMR Score Grade",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "<P25", 2: "P25-P50", 3: "P50-P75", 4: "≥P75"}[x]
        )

    # Model input dictionary (11 features, ETE removed)
    input_dict = {
        'New_focal': New_focal,
        'CDFI': CDFI,
        'SIRI_four': SIRI_four,
        'HT': HT,
        'TSH': TSH,
        'Tumor_size_custom': Tumor_size_custom,
        'NG': NG,
        'Boundary': Boundary,
        'Microcalcification': Microcalcification,
        'LMR_four': LMR_four,
        'Special_location': Special_location
    }
    return input_dict


def display_prediction_result(prediction, probability):
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = "🔴" if prediction == 1 else "🟢"
    st.markdown(f'<div class="sub-header">{risk_color} Prediction Result</div>', unsafe_allow_html=True)

    box_class = "prediction-box high-risk" if prediction == 1 else "prediction-box low-risk"
    st.markdown(f"""
    <div class="{box_class}">
        <h3 style="margin-top:0;">{risk_level}</h3>
        <p><strong>Lymph Node Metastasis Probability:</strong> {probability * 100:.1f}%</p>
        <p><strong>Clinical Recommendation:</strong> {"Further lymph node examination and intraoperative lymph node dissection recommended" if prediction == 1 else "Routine follow-up monitoring is sufficient"}</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(probability))
    st.caption(f"Lymph Node Metastasis Possibility: {probability * 100:.1f}%")
    if probability < 0.3:
        st.info("📊 **Risk Interpretation**: Low risk range")
    elif probability < 0.7:
        st.warning("📊 **Risk Interpretation**: Moderate risk range, close follow-up recommended")
    else:
        st.error("📊 **Risk Interpretation**: High risk range, strongly recommend further examination")


def display_feature_importance(predictor):
    if predictor.model_info and 'feature_importance' in predictor.model_info:
        st.markdown('<div class="sub-header">📊 Feature Importance Analysis</div>', unsafe_allow_html=True)
        importance_data = predictor.model_info['feature_importance']
        importance_df = pd.DataFrame({
            'Feature': list(importance_data.keys()),
            'Importance': list(importance_data.values())
        }).sort_values('Importance', ascending=True)

        # English feature name mapping (updated)
        feature_names = {
            'New_focal': 'New Lesion', 'CDFI': 'Color Doppler Flow', 'SIRI_four': 'SIRI Score',
            'HT': "Hashimoto's", 'TSH': 'TSH Level', 'Tumor_size_custom': 'Tumor Size',
            'NG': 'Nodular Goiter', 'Boundary': 'Boundary Clarity', 'Microcalcification': 'Microcalcification',
            'LMR_four': 'LMR Score', 'Special_location': 'Special Location'
        }
        importance_df['Feature_EN'] = importance_df['Feature'].map(feature_names)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['Importance'], color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature_EN'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Model Feature Importance Ranking')
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)


def main():
    st.markdown('<div class="main-header">🏥 Thyroid Cancer Lymph Node Metastasis Prediction System</div>',
                unsafe_allow_html=True)
    predictor = ThyroidCancerPredictor()

    tab1, tab2, tab3 = st.tabs(["🔍 Risk Prediction", "📈 Model Analysis", "ℹ️ About"])

    with tab1:
        st.markdown("### Please enter patient clinical information for lymph node metastasis risk assessment")
        input_data = create_input_form()

        with st.expander("📖 Feature Description"):
            st.markdown("""
            - **New Lesion Type**: 1=Unilateral single-focus, 2=Unilateral multifocal, 3=Bilateral lesions
            - **Color Doppler Flow Signal**: Reflects tumor blood supply (0=No, 1=Yes)
            - **Boundary Clarity**: Whether the tumor boundary is clear (0=Clear, 1=Unclear)
            - **Special Tumor Location**: 0=Pole & Isthmus, 1=Near Trachea, 2=Middle (Thyroid gland center)
            - **Nodular Goiter**: Whether complicated with nodular goiter (0=No, 1=Yes)
            - **Hashimoto's Thyroiditis**: Autoimmune thyroiditis (0=No, 1=Yes)
            - **Microcalcification**: Whether microcalcification is found on ultrasound (0=No, 1=Yes)
            - **Tumor Size**: 1=<10mm, 2=10-20mm, 3=≥20mm
            - **TSH Level**: Thyroid Stimulating Hormone level (mU/L)
            - **SIRI Score**: Systemic Inflammation Response Index grade (1-4)
            - **LMR Score**: Lymphocyte to Monocyte Ratio grade (1-4)
            """)

        if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
            st.markdown("### 📋 Input Data Summary")
            summary_data = {
                'Feature': ['New Lesion', 'Color Doppler', 'SIRI Score', "Hashimoto's", 'TSH',
                            'Tumor Size', 'Nodular Goiter', 'Boundary Clarity', 'Microcalcification',
                            'LMR Score', 'Special Location'],
                'Value': [
                    {1: 'Unilateral single-focus', 2: 'Unilateral multifocal', 3: 'Bilateral lesions'}[
                        input_data['New_focal']],
                    'Yes' if input_data['CDFI'] == 1 else 'No',
                    f"Grade {input_data['SIRI_four']}",
                    'Yes' if input_data['HT'] == 1 else 'No',
                    f"{input_data['TSH']} mU/L",
                    {1: '<10mm', 2: '10-20mm', 3: '≥20mm'}[input_data['Tumor_size_custom']],
                    'Yes' if input_data['NG'] == 1 else 'No',
                    'Clear' if input_data['Boundary'] == 0 else 'Unclear',
                    'Yes' if input_data['Microcalcification'] == 1 else 'No',
                    f"Grade {input_data['LMR_four']}",
                    {0: 'Pole & Isthmus', 1: 'Near Trachea', 2: 'Middle'}[input_data['Special_location']]
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            with st.spinner('Analyzing...'):
                prediction, probability = predictor.predict(input_data)
            if prediction is not None:
                display_prediction_result(prediction, probability)

    with tab2:
        st.markdown("### Model Performance Analysis")
        if predictor.model_info:
            metrics = predictor.model_info.get('metrics', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
            with col2:
                st.metric("Sensitivity", f"{metrics.get('sensitivity', 0):.3f}")
                st.metric("Specificity", f"{metrics.get('specificity', 0):.3f}")
            st.info(f"**Number of Features:** {len(predictor.model_info['features'])}")
            display_feature_importance(predictor)
        else:
            st.warning("Model information not loaded. Train the model first.")

    with tab3:
        st.markdown("### About This System")
        st.write("""
        This system predicts lymph node metastasis risk in thyroid cancer patients using a CatBoost machine learning model.
        It integrates ultrasound, pathological, and laboratory indicators to assist in clinical decision-making.
        """)
        st.caption("For research use. Clinical decisions should not be based solely on this tool.")


if __name__ == "__main__":
    main()