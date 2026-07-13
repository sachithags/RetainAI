import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import os

# ─── Global references (loaded lazily) ───
_model = None
_preprocessor = None
_policies = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("model.pkl")
    return _model

def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = joblib.load("preprocessor.pkl")
    return _preprocessor

def get_policies():
    global _policies
    if _policies is None:
        with open("policy_db.json", "r") as f:
            _policies = json.load(f)
    return _policies

# ─── CSS ───
custom_css = """
#result-card {
    border-radius: 16px; padding: 22px 26px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
footer { display: none !important; }
"""

# ─── Premium Result Card ───
def render_result_card(proba):
    is_high = proba >= 0.5
    color = "#ef4444" if is_high else "#22c55e"
    label = "HIGH RISK" if is_high else "LOW RISK"
    icon = "🚨" if is_high else "✅"
    return f"""
    <div id="result-card" style="background:#111827;border-left:6px solid {color};">
      <div style="font-size:13px;letter-spacing:1.5px;color:#9ca3af;">ATTRITION RISK</div>
      <div style="font-size:26px;font-weight:700;color:{color};margin:6px 0;">{icon} {label}</div>
      <div style="font-size:46px;font-weight:800;color:white;">{proba:.0%}</div>
    </div>
    """

# ─── Deterministic policy engine ───
def match_policies(emp):
    policies = get_policies()
    matched = []
    for p in policies:
        cond = p["conditions"]
        ok = True
        for key, val in cond.items():
            if key in emp:
                if isinstance(val, list):
                    if emp[key] not in val:
                        ok = False; break
                else:
                    if emp[key] < val:
                        ok = False; break
        if ok:
            matched.append(p["policy_text"])
    return matched

def format_memo(employee_name, risk_prob, policies):
    if not policies:
        return "No specific retention policies found for this employee."
    lines = [
        f"## Retention Plan for {employee_name}",
        f"**Attrition risk: {risk_prob:.0%}**",
        "**Recommended actions:**"
    ]
    for p in policies:
        lines.append(f"- {p}")
    lines.append("\n*These suggestions are based on current company policy. Please review before taking action.*")
    return "\n".join(lines)

# ─── Prediction function ───
def predict_attrition(
    Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
    Education, EducationField, EmployeeCount, EnvironmentSatisfaction,
    Gender, HourlyRate, JobInvolvement, JobLevel, JobRole,
    JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
    NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
    RelationshipSatisfaction, StandardHours, StockOptionLevel,
    TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
    YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
    YearsWithCurrManager
):
    model = get_model()
    preprocessor = get_preprocessor()

    emp = {
        "Age": Age, "BusinessTravel": BusinessTravel, "DailyRate": DailyRate,
        "Department": Department, "DistanceFromHome": DistanceFromHome,
        "Education": Education, "EducationField": EducationField,
        "EmployeeCount": EmployeeCount, "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender, "HourlyRate": HourlyRate, "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel, "JobRole": JobRole, "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus, "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate, "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime, "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating, "RelationshipSatisfaction": RelationshipSatisfaction,
        "StandardHours": StandardHours, "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }
    emp["Over18"] = "Y"

    df = pd.DataFrame([emp])
    transformed = preprocessor.transform(df)
    proba = model.predict_proba(transformed)[0][1]

    risk_msg = render_result_card(proba)

    if proba >= 0.5:
        policies = match_policies(emp)
        memo = format_memo("Employee", proba, policies)
    else:
        memo = "No retention plan needed (low risk)."

    return risk_msg, memo

# ─── Sample employee data ───
def load_low_risk():
    return [35, "Travel_Rarely", 800, "Research & Development", 3, 4, "Life Sciences", 1, 4,
            "Male", 70, 3, 2, "Research Scientist", 4, "Married", 12000, 15000,
            1, "No", 15, 4, 4, 80, 2, 12, 2, 4, 8, 4, 3, 5]

def load_high_risk():
    return [30, "Travel_Rarely", 500, "Sales", 5, 1, "Life Sciences", 1, 1,
            "Male", 50, 3, 1, "Sales Executive", 3, "Single", 5000, 10000,
            2, "Yes", 12, 3, 3, 80, 0, 10, 3, 3, 5, 2, 1, 2]

def load_borderline():
    return [42, "Travel_Frequently", 1100, "Sales", 9, 3, "Marketing", 1, 2,
            "Female", 85, 3, 3, "Manager", 2, "Married", 9000, 12000,
            3, "Yes", 14, 3, 2, 80, 1, 18, 4, 1, 10, 7, 1, 8]

def load_random():
    return [28, "Non-Travel", 600, "Human Resources", 12, 2, "Human Resources", 1, 3,
            "Female", 40, 2, 1, "Human Resources", 3, "Divorced", 4500, 8000,
            1, "No", 10, 3, 3, 80, 0, 6, 3, 3, 2, 2, 0, 2]

# ─── RAG tab function ───
def rag_answer(query):
    if not query.strip():
        return "Please enter a question.", ""

    try:
        from groq import Groq
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        return f"RAG dependencies missing: {e}", ""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set in Space secrets. RAG tab is disabled.", ""

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    client = Groq(api_key=api_key)
    prompt = f"""You are an HR policy assistant. Answer ONLY using the context below.
If the answer isn't in the context, say you don't have that information — do not guess.

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer, context

# ─── Theme ───
theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_text_color="*neutral_800",
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="*shadow_sm",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="white",
    input_background_fill="white",
    input_border_color="*neutral_300",
    input_shadow="*shadow_xs",
    slider_color="*primary_500",
    checkbox_label_text_color="*neutral_700",
    link_text_color="*primary_600",
)

# ─── Gradio UI ───
with gr.Blocks(title="RetainAI") as demo:
    gr.Markdown("# 🔮 RetainAI — Employee Attrition Predictor & Retention Planner")
    gr.Markdown("Analyse employee data to identify flight risks and get policy‑based retention suggestions. Use the **Policy Q&A** tab to ask about company policies.")

    with gr.Tabs():
        # ── Prediction Tab ──
        with gr.TabItem("Attrition Prediction"):
            with gr.Row():
                sample_low = gr.Button("👤 Low Risk")
                sample_high = gr.Button("👤 High Risk")
                sample_border = gr.Button("👤 Borderline")
                sample_random = gr.Button("🎲 Random")

            with gr.Tabs():
                with gr.TabItem("Personal Details"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            Age = gr.Number(label="Age", value=30)
                            Gender = gr.Dropdown(["Male","Female"], label="Gender")
                            MaritalStatus = gr.Dropdown(["Single","Married","Divorced"], label="Marital Status")
                            Education = gr.Dropdown([1,2,3,4,5], label="Education")
                            EducationField = gr.Dropdown(["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"], label="Education Field")
                        with gr.Column(scale=1):
                            Department = gr.Dropdown(["Sales","Research & Development","Human Resources"], label="Department")
                            JobRole = gr.Dropdown(["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"], label="Job Role")
                            JobLevel = gr.Dropdown([1,2,3,4,5], label="Job Level")
                            NumCompaniesWorked = gr.Number(label="Num Companies Worked", value=2)
                            DistanceFromHome = gr.Number(label="Distance From Home", value=5)

                with gr.TabItem("Work & Performance"):
                    with gr.Row():
                        with gr.Column():
                            JobInvolvement = gr.Slider(1,4,value=3, label="Job Involvement")
                            JobSatisfaction = gr.Slider(1,4,value=3, label="Job Satisfaction")
                            PerformanceRating = gr.Dropdown([3,4], label="Performance Rating")
                            WorkLifeBalance = gr.Slider(1,4,value=3, label="Work Life Balance")
                            RelationshipSatisfaction = gr.Slider(1,4,value=3, label="Relationship Satisfaction")
                        with gr.Column():
                            EnvironmentSatisfaction = gr.Slider(1,4,value=3, label="Environment Satisfaction")
                            OverTime = gr.Dropdown(["Yes","No"], label="Over Time")
                            YearsAtCompany = gr.Number(label="Years At Company", value=5)
                            YearsInCurrentRole = gr.Number(label="Years In Current Role", value=2)
                            YearsSinceLastPromotion = gr.Number(label="Years Since Last Promotion", value=1)
                            YearsWithCurrManager = gr.Number(label="Years With Current Manager", value=2)

                with gr.TabItem("Compensation"):
                    with gr.Row():
                        with gr.Column():
                            MonthlyIncome = gr.Number(label="Monthly Income", value=5000)
                            MonthlyRate = gr.Number(label="Monthly Rate", value=10000)
                            DailyRate = gr.Number(label="Daily Rate", value=500)
                            HourlyRate = gr.Number(label="Hourly Rate", value=50)
                        with gr.Column():
                            PercentSalaryHike = gr.Number(label="Percent Salary Hike", value=12)
                            StockOptionLevel = gr.Dropdown([0,1,2,3], label="Stock Option Level")
                            EmployeeCount = gr.Number(label="Employee Count", value=1)
                            StandardHours = gr.Number(label="Standard Hours", value=80)

                with gr.TabItem("Other"):
                    with gr.Accordion("Less common fields", open=False):
                        BusinessTravel = gr.Dropdown(["Travel_Rarely","Travel_Frequently","Non-Travel"], label="Business Travel")
                        TotalWorkingYears = gr.Number(label="Total Working Years", value=10)
                        TrainingTimesLastYear = gr.Number(label="Training Times Last Year", value=3)

            with gr.Row():
                btn = gr.Button("🔍 Predict Attrition Risk", variant="primary")

            with gr.Row():
                risk_output = gr.HTML(label="Attrition Risk")
                plan_output = gr.Markdown(label="Retention Plan")

            all_inputs = [Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                          Education, EducationField, EmployeeCount, EnvironmentSatisfaction,
                          Gender, HourlyRate, JobInvolvement, JobLevel, JobRole,
                          JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
                          NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
                          RelationshipSatisfaction, StandardHours, StockOptionLevel,
                          TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
                          YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
                          YearsWithCurrManager]
            sample_low.click(fn=load_low_risk, outputs=all_inputs)
            sample_high.click(fn=load_high_risk, outputs=all_inputs)
            sample_border.click(fn=load_borderline, outputs=all_inputs)
            sample_random.click(fn=load_random, outputs=all_inputs)

            btn.click(fn=predict_attrition, inputs=all_inputs, outputs=[risk_output, plan_output])

        # ── RAG Tab ──
        with gr.TabItem("Policy Q&A (RAG)"):
            gr.Markdown("### Ask about company policies — answers are grounded in real documents, never made up.")
            query_box = gr.Textbox(label="Your question", placeholder="e.g., What is the parental leave policy?")
            answer_box = gr.Textbox(label="Answer", lines=6)
            sources_box = gr.Textbox(label="Retrieved context (for transparency)", lines=4)
            ask_btn = gr.Button("Ask")
            ask_btn.click(fn=rag_answer, inputs=query_box, outputs=[answer_box, sources_box])

demo.launch(server_name="0.0.0.0", theme=theme, css=custom_css)