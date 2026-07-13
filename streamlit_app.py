import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="RetainAI", page_icon="🔮", layout="wide")

st.title("🔮 RetainAI – Employee Attrition Risk Predictor")
st.markdown("Upload a CSV file or fill in employee details manually to predict who is likely to leave.")

API_URL = "http://127.0.0.1:8000/predict"

# Tabs for different modes
tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Manual Input"])

with tab1:
    st.header("Upload Employee Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        if st.button("Predict for all employees"):
            results = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    resp = requests.post(API_URL, json=row.to_dict())
                    if resp.status_code == 200:
                        res_json = resp.json()
                        results.append({
                            **row.to_dict(),
                            "Attrition Risk": "Yes" if res_json["attrition"] else "No",
                            "Probability": res_json["probability"]
                        })
                    else:
                        results.append({**row.to_dict(), "Error": "API call failed"})
                except Exception as e:
                    results.append({**row.to_dict(), "Error": str(e)})
                progress.progress((i+1)/len(df))
            result_df = pd.DataFrame(results)
            st.success(f"Predictions completed for {len(results)} employees.")
            st.dataframe(result_df)
            st.download_button("Download Predictions", result_df.to_csv(index=False), "retainai_predictions.csv")

with tab2:
    st.header("Enter Employee Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        daily_rate = st.number_input("Daily Rate", value=500)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        distance = st.number_input("Distance From Home", value=5)
        education = st.selectbox("Education Level", [1,2,3,4,5])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        employee_count = st.number_input("Employee Count", value=1)
        env_satisfaction = st.slider("Environment Satisfaction", 1,4,3)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        hourly_rate = st.number_input("Hourly Rate", value=50)
        job_involvement = st.slider("Job Involvement", 1,4,3)
        job_level = st.selectbox("Job Level", [1,2,3,4,5])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        job_satisfaction = st.slider("Job Satisfaction", 1,4,3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        monthly_income = st.number_input("Monthly Income", value=5000)
        monthly_rate = st.number_input("Monthly Rate", value=10000)
        num_companies = st.number_input("Num Companies Worked", value=2)
        overtime = st.selectbox("Over Time", ["Yes", "No"])
    with col3:
        percent_hike = st.number_input("Percent Salary Hike", value=12)
        performance = st.selectbox("Performance Rating", [3,4])
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1,4,3)
        standard_hours = st.number_input("Standard Hours", value=80)
        stock_option = st.selectbox("Stock Option Level", [0,1,2,3])
        total_working_years = st.number_input("Total Working Years", value=10)
        training_last_year = st.number_input("Training Times Last Year", value=3)
        work_life_balance = st.slider("Work Life Balance", 1,4,3)
        years_at_company = st.number_input("Years At Company", value=5)
        years_in_role = st.number_input("Years In Current Role", value=2)
        years_since_promotion = st.number_input("Years Since Last Promotion", value=1)
        years_with_manager = st.number_input("Years With Current Manager", value=2)

    if st.button("Predict Attrition Risk"):
        payload = {
            "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
            "Department": department, "DistanceFromHome": distance, "Education": education,
            "EducationField": education_field, "EmployeeCount": employee_count,
            "EnvironmentSatisfaction": env_satisfaction, "Gender": gender, "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement, "JobLevel": job_level, "JobRole": job_role,
            "JobSatisfaction": job_satisfaction, "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "OverTime": overtime,
            "PercentSalaryHike": percent_hike, "PerformanceRating": performance,
            "RelationshipSatisfaction": relationship_satisfaction, "StandardHours": standard_hours,
            "StockOptionLevel": stock_option, "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_last_year, "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion, "YearsWithCurrManager": years_with_manager
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                if result["attrition"]:
                    st.error(f"⚠️ High risk of attrition (probability: {result['probability']})")
                else:
                    st.success(f"✅ Low risk of attrition (probability: {result['probability']})")
            else:
                st.warning("Prediction failed. Please check the API.")
        except Exception as e:
            st.error(f"Connection error: {e}")