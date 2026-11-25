import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import numpy as np
import streamlit.components.v1 as components

# --- Load model ---
# Ensure the path to your model is correct
try:
    heart_disease_model = pickle.load(open(r"D:\Research work\Mplementation\Heart Disease Prediction\heart_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please check the path and ensure the model is available.")
    # Create a dummy model for local testing if the actual model isn't available
    class DummyModel:
        def predict(self, df):
            # Simulate a prediction (0 for no disease, 1 for disease)
            if 'age' in df.columns and df['age'].iloc[0] > 55 and df['thalach'].iloc[0] < 120:
                return [1] # Older with low max heart rate is more likely to have disease
            return [0]

        def predict_proba(self, df):
            # Simulate probabilities
            if 'age' in df.columns and df['age'].iloc[0] > 55 and df['thalach'].iloc[0] < 120:
                return np.array([[0.25, 0.75]]) # 75% chance of disease
            return np.array([[0.85, 0.15]]) # 15% chance of disease
    heart_disease_model = DummyModel()


# --- Helper function to show SHAP in Streamlit ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- MODIFIED: Function to create dynamic donut chart ---
def create_donut_chart(percentage, label, has_disease=False):
    """
    Creates a donut chart with conditional coloring.
    - Green for 'No Disease'.
    - Red for 'Disease'.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # Make the figure and axes backgrounds transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set colors based on the prediction
    if has_disease:
        main_color = '#d9534f'      # Red for disease
        secondary_color = '#a94442'  # Darker red for the rest of the circle
    else:
        main_color = '#5cb85c'      # Green for no disease
        secondary_color = '#3e8e41'  # Darker green for the rest of the circle

    # Create the pie chart
    size_of_chunks = [percentage, 100 - percentage]
    ax.pie(size_of_chunks, colors=[main_color, secondary_color], wedgeprops=dict(width=0.3), startangle=90)

    # Add a circle in the center to create the donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='black')
    fig.gca().add_artist(centre_circle)

    # Add the percentage text in the center
    ax.text(0, 0, f"{percentage:.2f}%", ha='center', va='center', fontsize=22, color='white', weight='bold')

    # Add the label at the top with a colored background
    fig.suptitle(label, fontsize=16, color='white', y=0.98, backgroundcolor=main_color, x=0.5, ha='center', weight='bold')

    ax.axis('equal')  # Equal aspect ratio ensures the pie is a circle.
    return fig

def main():
    st.set_page_config(page_title="Explainable AI Based Multi Disease Prediction", layout="wide")
    st.title("Explainable AI Based Multi Disease Prediction And Recommendation System")

    # --- Sidebar ---
    with st.sidebar:
        try:
            image = Image.open(r"D:\Research work\Mplementation\image\logo.png")
            st.image(image, width=300)
        except FileNotFoundError:
            st.warning("Logo image not found. Please check the path.")
            st.image("https://via.placeholder.com/300x150.png?text=Logo", caption="Placeholder Logo", width=300)

        selected = option_menu(
            'Explainable AI System',
            ["Heart Disease Prediction", "Diabetes Prediction", "General Health Prediction", "Understand the Input Features"],
            icons=['heart', 'activity', 'person', 'activity'],
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#304F7E"},
                "icon": {"color": "black", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#1c1b1b"
                },
                "nav-link-selected": {"background-color": "#007bff"}
            }
        )

    # --- Heart Disease Page ---
    if selected == "Heart Disease Prediction":
        st.header("Heart Disease Prediction")
        col1, col2, col3, col4 = st.columns(4)

        # Input fields
        with col1: age = st.number_input("Age", min_value=1, max_value=120, value=50)
        with col2:
            sex_options = ['Select', 'Male', 'Female']
            sex = st.selectbox('Sex', sex_options)
        with col3: cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
        with col1: trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=220)
        with col2: chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
        with col3: fbs = st.number_input('Fasting Blood Sugar > 120 (1=True, 0=False)', min_value=0, max_value=1)
        with col1: restecg = st.number_input('Resting ECG (0,1,2)', min_value=0, max_value=2)
        with col2: thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220)
        with col3: exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0, max_value=1)
        with col1: oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        with col2: slope = st.number_input('Slope (0,1,2)', min_value=0, max_value=2)
        with col3: ca = st.number_input('Major Vessels (0-3)', min_value=0, max_value=3)
        with col1: thal = st.number_input('Thalassemia (1,2,3)', min_value=1, max_value=3)

        with col4:
            try:
                image = Image.open(r"D:\Research work\Mplementation\image\heart.png")
                st.image(image, width=300)
            except FileNotFoundError:
                st.warning("Heart image not found. Please check the path.")
                st.image("https://via.placeholder.com/300x150.png?text=Heart", caption="Placeholder Heart", width=300)

        # --- Prediction Trigger ---
        if st.button('Analyze Heart Health'):
            if sex == 'Select':
                st.error("Please select a valid gender.")
            else:
                try:
                    sex_mapping = {'Male': 1, 'Female': 0}
                    sex_numeric = sex_mapping[sex]

                    input_data = [float(age), float(sex_numeric), float(cp), float(trestbps), float(chol), float(fbs),
                                  float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                                  float(ca), float(thal)]
                    
                    input_df = pd.DataFrame([input_data],
                                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

                    # --- Prediction ---
                    prediction = heart_disease_model.predict(input_df)[0]
                    proba = heart_disease_model.predict_proba(input_df)[0]

                    st.subheader("Prediction Result")
                    
                    col_donut, col_metrics = st.columns([1, 2])
                    
                    # --- MODIFIED: Dynamic Donut Chart Logic ---
                    with col_donut:
                        if prediction == 1:  # Has disease
                            percentage = proba[1] * 100
                            label = "Heart Disease"
                            has_disease = True
                        else:  # No disease
                            percentage = proba[0] * 100
                            label = "No Heart Disease"
                            has_disease = False
                        
                        # Call the updated function
                        donut_fig = create_donut_chart(percentage, label, has_disease=has_disease)
                        st.pyplot(donut_fig, use_container_width=True)
                        plt.clf() # Clear the plot to prevent memory leaks

                    with col_metrics:
                        st.metric("Probability of No Heart Disease", f"{proba[0] * 100:.2f}%")
                        st.metric("Probability of Heart Disease", f"{proba[1] * 100:.2f}%")

                        if prediction == 1:
                            st.error('The model predicts that the person **has heart disease**.')
                            with st.expander("Recommendation:"):
                                st.warning("It is highly recommended to consult a doctor immediately. Based on the model, lifestyle changes and specific medications like beta-blockers or statins may be advised by a healthcare professional.")
                        else:
                            st.success('The model predicts that the person **does not have heart disease**.')
                            with st.expander("Recommendation:"):
                                st.info("Maintain a healthy lifestyle with a balanced diet and regular exercise. Continue with regular health check-ups.")
                    
                    st.markdown("---")
                    
                    # --- SHAP Explanation ---
                    st.subheader("Explainable AI (SHAP) Visualization")
                    explainer = shap.TreeExplainer(heart_disease_model)
                    shap_values = explainer.shap_values(input_df)

                    if isinstance(shap_values, list):
                        shap_array = np.array(shap_values[1]).reshape(1, -1)
                        expected_value = explainer.expected_value[1]
                    else:
                        shap_array = np.array(shap_values).reshape(1, -1)
                        expected_value = explainer.expected_value

                    if isinstance(expected_value, (list, np.ndarray)):
                        base_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]
                    else:
                        base_value = float(expected_value)

                    num_features = len(input_df.columns)
                    flat_values = shap_array.flatten()[:num_features]

                    shap_explanation = shap.Explanation(
                        values=flat_values,
                        base_values=base_value,
                        data=input_df.iloc[0],
                        feature_names=input_df.columns.tolist()
                    )
                    
                    # Store SHAP Explanation in session state
                    st.session_state["shap_explanation"] = shap_explanation
                    st.session_state["prediction"] = prediction

                    # SHAP Waterfall Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_explanation, show=False)
                    st.pyplot(fig)
                    plt.clf()

                    # Text Explanation based on SHAP values
                    st.subheader("How Features Influence the Prediction")
                    if "shap_explanation" in st.session_state:
                        exp = st.session_state["shap_explanation"]
                        prediction_label = 'Heart Disease' if st.session_state["prediction"] == 1 else 'No Heart Disease'
                        
                        shap_df = pd.DataFrame({
                            'Feature': exp.feature_names,
                            'SHAP Value': exp.values,
                            'Feature Value': exp.data
                        })
                        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
                        shap_df = shap_df.sort_values(by='Abs SHAP', ascending=False)
                        
                        st.markdown(f"The model predicted **{prediction_label}**. Here's a breakdown of the most influential factors:")
                        
                        positive_impact_features = shap_df[shap_df['SHAP Value'] > 0]
                        if not positive_impact_features.empty:
                            st.markdown("#### Features Increasing the Likelihood of Heart Disease:")
                            for _, row in positive_impact_features.head(3).iterrows(): # Show top 3
                                st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This was a significant factor pushing the prediction towards disease.")
                        
                        negative_impact_features = shap_df[shap_df['SHAP Value'] < 0]
                        if not negative_impact_features.empty:
                            st.markdown("#### Features Decreasing the Likelihood of Heart Disease:")
                            for _, row in negative_impact_features.head(3).iterrows(): # Show top 3
                                st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This factor pushed the prediction away from disease.")

                except Exception as e:
                    st.error(f"An error occurred. Please ensure all inputs are valid numbers. Error: {e}")

if __name__ == "__main__":
    main()
    