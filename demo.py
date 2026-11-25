import streamlit as st
import pickle
import pandas as pd
import PyPDF2
import re
import os
from fpdf import FPDF

# --- DEFAULT VALUES for missing PDF data ---
# These values are often derived from the mean/median of the training dataset
DEFAULT_VALUES = {
    'age': 54, 'sex': 1, 'cp': 0, 'trestbps': 131, 'chol': 246,
    'fbs': 0, 'restecg': 0, 'thalach': 149, 'exang': 0,
    'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
}

# --- PDF REPORT GENERATION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Heart Disease Prediction Report', 1, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, data):
        self.set_font('Arial', '', 12)
        for key, value in data.items():
            self.cell(0, 10, f"{key.replace('_', ' ').title()}: {value}", 0, 1)
        self.ln()

    def prediction_result(self, result, color):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(color[0], color[1], color[2])
        self.cell(0, 10, f"Prediction Result: {result}", 0, 1, 'C')
        self.set_text_color(0, 0, 0)


def generate_pdf_report(input_data, prediction_text):
    """Generates a downloadable PDF report."""
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title('Patient Input Parameters')
    pdf.chapter_body(input_data)
    
    result_color = (217, 83, 79) if "May Have" in prediction_text else (92, 184, 92) # Red or Green
    pdf.prediction_result(prediction_text, result_color)
    
    return pdf.output(dest='S').encode('latin-1')


# --- LOAD THE PRE-TRAINED MODEL ---
def load_model(model_path="D:\Research work\Mplementation\Heart Disease Prediction\heart_model.pkl"):
    """Loads the pre-trained model from a pickle file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- PDF TEXT EXTRACTION AND PARSING ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")
        return ""

def parse_pdf_data(text):
    """Parses extracted text to find medical parameters using regex."""
    data = {}
    patterns = {
        'age': r"Age\s*:\s*(\d+)",
        'sex': r"Sex\s*:\s*(Male|Female)",
        'cp': r"Chest Pain Type\s*:\s*(\d)",
        'trestbps': r"Resting Blood Pressure\s*:\s*(\d+)",
        'chol': r"Cholesterol\s*:\s*(\d+)",
        'fbs': r"Fasting Blood Sugar\s*>\s*120\s*mg/dl\s*:\s*(0|1|True|False)",
        'restecg': r"Resting ECG\s*:\s*(\d)",
        'thalach': r"Max Heart Rate\s*:\s*(\d+)",
        'exang': r"Exercise Induced Angina\s*:\s*(0|1|Yes|No)",
        'oldpeak': r"ST Depression\s*:\s*(\d\.\d+|\d+)",
        'slope': r"Slope\s*:\s*(\d)",
        'ca': r"Major Vessels\s*:\s*(\d)",
        'thal': r"Thalassemia\s*:\s*(\d)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            # Convert text values to numeric
            if key == 'sex':
                data[key] = 1 if value == 'male' else 0
            elif key == 'fbs':
                data[key] = 1 if value in ['1', 'true'] else 0
            elif key == 'exang':
                data[key] = 1 if value in ['1', 'yes'] else 0
            else:
                data[key] = float(value)
    return data

# --- MAIN STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

    st.title("❤️ Heart Disease Prediction System")
    st.subheader("Predict Heart Disease using Manual Entry or PDF Report")

    # Load the model
    heart_disease_model = load_model()
    if heart_disease_model is None:
        st.stop() # Stop the app if the model can't be loaded

    # Sidebar for input method selection
    st.sidebar.title("Navigation")
    input_method = st.sidebar.radio("Choose your input method:", ("Enter Values Manually", "Upload PDF Report"))

    if input_method == "Enter Values Manually":
        st.header("Manual Input Parameters")
        
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, step=1)
                sex = st.selectbox("Sex", ("Male", "Female"))
                cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3))
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220)

            with col2:
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("False", "True"))
                restecg = st.selectbox("Resting Electrocardiographic Results", (0, 1, 2))
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)

            with col3:
                exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
                oldpeak = st.number_input("ST Depression induced by exercise", min_value=0.0, max_value=7.0, step=0.1)
                slope = st.selectbox("Slope of the peak exercise ST segment", (0, 1, 2))
                ca = st.selectbox("Number of major vessels colored by flourosopy", (0, 1, 2, 3))
                thal = st.selectbox("Thalassemia", (0, 1, 2, 3), help="0=None, 1=Normal, 2=Fixed defect, 3=Reversible defect")
            
            submit_button = st.form_submit_button(label="🩺 Predict Heart Disease")

            if submit_button:
                # Convert inputs to the model's required format
                sex_numeric = 1 if sex == "Male" else 0
                fbs_numeric = 1 if fbs == "True" else 0
                exang_numeric = 1 if exang == "Yes" else 0
                
                input_data = {
                    'age': age, 'sex': sex_numeric, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                    'fbs': fbs_numeric, 'restecg': restecg, 'thalach': thalach, 'exang': exang_numeric,
                    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                }
                
                # Display summary table
                st.subheader("Summary of Input Parameters")
                st.dataframe(pd.DataFrame([input_data]))

                # Prediction
                prediction_df = pd.DataFrame([input_data])
                prediction = heart_disease_model.predict(prediction_df)
                
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    prediction_text = "The person may have Heart Disease."
                    st.error(prediction_text)
                else:
                    prediction_text = "The person is likely Healthy."
                    st.success(prediction_text)
                
                # Download Report Button
                pdf_report = generate_pdf_report(input_data, prediction_text)
                st.download_button(
                    label="📄 Download Report",
                    data=pdf_report,
                    file_name="heart_disease_report.pdf",
                    mime="application/pdf"
                )

    elif input_method == "Upload PDF Report":
        st.header("Upload Medical Report")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            extracted_data = parse_pdf_data(text)
            
            # Combine extracted data with defaults for missing values
            final_data = DEFAULT_VALUES.copy()
            final_data.update(extracted_data)
            
            st.subheader("Extracted & Filled Parameters")
            st.info("Values extracted from the PDF are shown below. Missing values have been filled with defaults.")
            st.dataframe(pd.DataFrame([final_data]))
            
            if st.button("🩺 Predict from PDF"):
                prediction_df = pd.DataFrame([final_data])
                prediction = heart_disease_model.predict(prediction_df)
                
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    prediction_text = "Based on the report, the person may have Heart Disease."
                    st.error(prediction_text)
                else:
                    prediction_text = "Based on the report, the person is likely Healthy."
                    st.success(prediction_text)

                # Download Report Button
                pdf_report = generate_pdf_report(final_data, prediction_text)
                st.download_button(
                    label="📄 Download Report",
                    data=pdf_report,
                    file_name="heart_disease_report_from_pdf.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
