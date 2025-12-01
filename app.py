# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Thyroid Cancer Lymph Node Metastasis Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .feature-info {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class ThyroidCancerPredictor:
    def __init__(self):
        self.model = None
        self.model_info = None
        self.features = []
        self.load_model()

    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('models/catboost_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            with open('models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)

            self.features = self.model_info['features']
            st.sidebar.success("✅ Model loaded successfully!")

        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("Please run train_model.py first to train the model")

    def predict(self, input_data):
        """Make prediction using the model"""
        if self.model is None:
            st.error("Model not loaded, cannot make prediction")
            return None, None

        try:
            # Ensure input data has same feature order as training
            input_df = pd.DataFrame([input_data])[self.features]

            # Predict probability
            probability = self.model.predict_proba(input_df)[0, 1]
            prediction = 1 if probability > 0.5 else 0

            return prediction, probability

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None, None


def create_input_form():
    """Create user input form"""
    st.markdown('<div class="sub-header">📋 Patient Clinical Features Input</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Ultrasound Features")
        New_focal = st.selectbox(
            "New Lesion Type",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Unilateral single-focus", 2: "Unilateral multifocal", 3: "Bilateral lesions"}[x],
            help="1=Unilateral single-focus, 2=Unilateral multifocal, 3=Bilateral lesions"
        )

        CDFI = st.selectbox(
            "Color Doppler Flow Signal",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="0=No, 1=Yes"
        )

        Boundary = st.selectbox(
            "Boundary Clarity",
            options=[0, 1],
            format_func=lambda x: "Clear" if x == 0 else "Unclear",
            help="0=Clear, 1=Unclear"
        )

    with col2:
        st.markdown("#### Pathological Features")
        NG = st.selectbox(
            "Nodular Goiter",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="0=No, 1=Yes"
        )

        ETE = st.selectbox(
            "Extra-thyroidal Extension",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="0=No, 1=Yes"
        )

        Microcalcification = st.selectbox(
            "Microcalcification",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="0=No, 1=Yes"
        )

        Tumor_size_custom = st.selectbox(
            "Tumor Size",
            options=[1, 2, 3],
            format_func=lambda x: {1: "<10mm", 2: "10-20mm", 3: "≥20mm"}[x],
            help="1=<10mm, 2=10-20mm, 3=≥20mm"
        )

    with col3:
        st.markdown("#### Laboratory Indicators")
        TSH = st.number_input(
            "TSH Level (mU/L)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Thyroid Stimulating Hormone level"
        )

        SIRI_four = st.selectbox(
            "SIRI Score Grade",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "<P25", 2: "P25-P50", 3: "P50-P75", 4: "≥P75"}[x],
            help="1=<P25, 2=P25-P50, 3=P50-P75, 4=≥P75"
        )

        LMR_four = st.selectbox(
            "LMR Score Grade",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "<P25", 2: "P25-P50", 3: "P50-P75", 4: "≥P75"}[x],
            help="1=<P25, 2=P25-P50, 3=P50-P75, 4=≥P75"
        )

    # Convert to model input format
    input_dict = {
        'New_focal': New_focal,
        'CDFI': CDFI,
        'SIRI_four': SIRI_four,
        'ETE': ETE,
        'TSH': TSH,
        'Tumor_size_custom': Tumor_size_custom,
        'NG': NG,
        'Boundary': Boundary,
        'Microcalcification': Microcalcification,
        'LMR_four': LMR_four
    }

    return input_dict


def display_prediction_result(prediction, probability):
    """Display prediction results"""
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = "🔴" if prediction == 1 else "🟢"
    risk_percentage = probability * 100

    st.markdown(f'<div class="sub-header">{risk_color} Prediction Result</div>', unsafe_allow_html=True)

    # Create dynamic styled prediction box
    box_class = "prediction-box high-risk" if prediction == 1 else "prediction-box low-risk"

    st.markdown(f"""
    <div class="{box_class}">
        <h3 style="margin-top:0;">{risk_level}</h3>
        <p><strong>Lymph Node Metastasis Probability:</strong> {risk_percentage:.1f}%</p>
        <p><strong>Clinical Recommendation:</strong> {"Further lymph node examination and intraoperative lymph node dissection recommended" if prediction == 1 else "Routine follow-up monitoring is sufficient"}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display probability bar
    st.progress(float(probability))
    st.caption(f"Lymph Node Metastasis Possibility: {risk_percentage:.1f}%")

    # Risk interpretation
    if probability < 0.3:
        st.info("📊 **Risk Interpretation**: Low risk range, low possibility of lymph node metastasis")
    elif probability < 0.7:
        st.warning("📊 **Risk Interpretation**: Moderate risk range, close follow-up recommended")
    else:
        st.error("📊 **Risk Interpretation**: High risk range, strongly recommend further examination")


def display_feature_importance(predictor):
    """Display feature importance"""
    if predictor.model_info and 'feature_importance' in predictor.model_info:
        st.markdown('<div class="sub-header">📊 Feature Importance Analysis</div>', unsafe_allow_html=True)

        importance_data = predictor.model_info['feature_importance']
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(importance_data.keys()),
            'Importance': list(importance_data.values())
        }).sort_values('Importance', ascending=True)

        # Feature name mapping
        feature_names = {
            'New_focal': 'New Lesion',
            'CDFI': 'Color Doppler Flow',
            'SIRI_four': 'SIRI Score',
            'ETE': 'Extra-thyroidal Extension',
            'TSH': 'TSH Level',
            'Tumor_size_custom': 'Tumor Size',
            'NG': 'Nodular Goiter',
            'Boundary': 'Boundary Clarity',
            'Microcalcification': 'Microcalcification',
            'LMR_four': 'LMR Score'
        }

        importance_df['Feature_EN'] = importance_df['Feature'].map(feature_names)

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(importance_df))

        bars = ax.barh(y_pos, importance_df['Importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature_EN'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Model Feature Importance Ranking')

        # Add values on bars
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)


def display_model_info(predictor):
    """Display model information"""
    if predictor.model_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Model Information")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Test Set Accuracy", f"{predictor.model_info['accuracy']:.3f}")
        with col2:
            st.metric("Test Set AUC", f"{predictor.model_info['auc']:.3f}")

        # Display feature importance summary
        st.sidebar.markdown("#### Key Features")
        if 'feature_importance' in predictor.model_info:
            importance = predictor.model_info['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            for feature, score in top_features:
                feature_en = {
                    'New_focal': 'New Lesion', 'CDFI': 'Color Doppler', 'SIRI_four': 'SIRI Score',
                    'ETE': 'Extra-thyroidal Extension', 'TSH': 'TSH Level', 'Tumor_size_custom': 'Tumor Size',
                    'NG': 'Nodular Goiter', 'Boundary': 'Boundary', 'Microcalcification': 'Microcalcification',
                    'LMR_four': 'LMR Score'
                }.get(feature, feature)
                st.sidebar.write(f"• {feature_en}: {score:.2f}")


def display_feature_info():
    """Display feature descriptions"""
    with st.expander("📖 Feature Description"):
        st.markdown("""
        ### Clinical Feature Description

        - **New Lesion Type**: 
          - 1=Unilateral single-focus, 2=Unilateral multifocal, 3=Bilateral lesions
        - **Color Doppler Flow Signal**: Reflects tumor blood supply
        - **Boundary Clarity**: Whether the tumor boundary is clear
        - **Nodular Goiter**: Whether complicated with nodular goiter
        - **Extra-thyroidal Extension**: Whether the tumor invades extra-thyroidal tissue
        - **Microcalcification**: Whether microcalcification is found on ultrasound
        - **Tumor Size**: 
          - 1=<10mm, 2=10-20mm, 3=≥20mm
        - **TSH Level**: Thyroid Stimulating Hormone level (mU/L)
        - **SIRI Score**: Systemic Inflammation Response Index
        - **LMR Score**: Lymphocyte to Monocyte Ratio
        """)


def main():
    """Main function"""
    # Application title
    st.markdown('<div class="main-header">🏥 Thyroid Cancer Lymph Node Metastasis Prediction System</div>',
                unsafe_allow_html=True)

    # Initialize predictor
    predictor = ThyroidCancerPredictor()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Prediction", "📈 Model Analysis", "ℹ️ About System"])

    with tab1:
        st.markdown("### Please enter patient clinical information for lymph node metastasis risk assessment")

        # Create input form
        input_data = create_input_form()

        # Display feature description
        display_feature_info()

        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🚀 Start Prediction", type="primary", use_container_width=True)

        if predict_btn:
            # Display input data summary
            st.markdown("### 📋 Input Data Summary")
            summary_data = {
                'Feature': ['New Lesion', 'Color Doppler', 'SIRI Score', 'Extra-thyroidal Extension', 'TSH',
                            'Tumor Size', 'Nodular Goiter', 'Boundary Clarity', 'Microcalcification', 'LMR Score'],
                'Value': [
                    {1: 'Unilateral single-focus', 2: 'Unilateral multifocal', 3: 'Bilateral lesions'}[
                        input_data['New_focal']],
                    'Yes' if input_data['CDFI'] == 1 else 'No',
                    f"Grade {input_data['SIRI_four']}",
                    'Yes' if input_data['ETE'] == 1 else 'No',
                    f"{input_data['TSH']} mU/L",
                    {1: '<10mm', 2: '10-20mm', 3: '≥20mm'}[input_data['Tumor_size_custom']],
                    'Yes' if input_data['NG'] == 1 else 'No',
                    'Clear' if input_data['Boundary'] == 0 else 'Unclear',
                    'Yes' if input_data['Microcalcification'] == 1 else 'No',
                    f"Grade {input_data['LMR_four']}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Make prediction
            with st.spinner('Analyzing, please wait...'):
                prediction, probability = predictor.predict(input_data)

            if prediction is not None:
                display_prediction_result(prediction, probability)

    with tab2:
        st.markdown("### Model Performance Analysis")

        if predictor.model_info:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Model Performance Metrics")
                st.info(f"""
                - **Accuracy**: {predictor.model_info['accuracy']:.3f}
                - **AUC**: {predictor.model_info['auc']:.3f}
                - **Number of Features**: {len(predictor.features)}
                """)

            with col2:
                st.markdown("#### Clinical Significance")
                st.success("""
                This model is based on machine learning algorithms, integrating ultrasound features, 
                pathological features and laboratory indicators to provide quantitative assessment 
                of lymph node metastasis risk for clinicians, assisting in developing personalized treatment plans.
                """)

            # Display feature importance
            display_feature_importance(predictor)

        else:
            st.warning("Model information not loaded, please train the model first")

    with tab3:
        st.markdown("### About This System")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            #### System Introduction
            This system is based on CatBoost machine learning algorithm for predicting lymph node metastasis risk in thyroid cancer patients.
            It integrates multi-dimensional clinical data to provide scientific basis for preoperative assessment.

            #### Technical Features
            - 🎯 **Accurate Prediction**: Based on real clinical data training
            - 🔄 **Multi-feature Integration**: Combining ultrasound, pathology and laboratory indicators
            - 💻 **User Friendly**: Simple and intuitive operation interface
            - 📱 **Instant Results**: Real-time risk probability calculation

            #### Target Population
            - Patients diagnosed with thyroid cancer
            - Preoperative lymph node metastasis risk assessment
            - Patients requiring personalized treatment planning
            """)

        with col2:
            st.markdown("""
            #### Important Notes
            - This system's prediction results are for reference only
            - Final diagnosis should combine clinical examination
            - Recommended to use under physician guidance
            - Regular model updates to maintain accuracy

            #### Technical Support
            - Algorithm: CatBoost
            - Framework: Streamlit
            - Language: Python
            """)

    # Display model information
    display_model_info(predictor)


if __name__ == "__main__":
    main()