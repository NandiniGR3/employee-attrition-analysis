# =========================================================
# EMPLOYEE ATTRITION ANALYSIS DASHBOARD
# Developed for Data Analyst Internship Project
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📊 Employee Attrition Analysis Dashboard")
st.markdown(
    "### HR Analytics Dashboard for Employee Retention & Attrition Insights"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "Employee_Attrition_Project/data/cleaned_dataset.csv"
    )

df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "Employee_Attrition_Project/models/random_forest_model.pkl"
    )

model = load_model()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("📌 Dashboard Filters")

selected_department = st.sidebar.multiselect(
    "Select Department",
    options=df["Department"].unique(),
    default=df["Department"].unique()
)

filtered_df = df[
    df["Department"].isin(selected_department)
]

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

st.header("📊 Executive Summary")

total_employees = len(filtered_df)

attrition_rate = (
    (filtered_df["Attrition"] == "Yes").sum()
    / total_employees
) * 100

average_salary = filtered_df["MonthlyIncome"].mean()

average_age = filtered_df["Age"].mean()

overtime_percentage = (
    (filtered_df["OverTime"] == "Yes").sum()
    / total_employees
) * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Employees",
    total_employees
)

col2.metric(
    "Attrition Rate",
    f"{attrition_rate:.2f}%"
)

col3.metric(
    "Avg Salary",
    f"${average_salary:,.0f}"
)

col4.metric(
    "Avg Age",
    f"{average_age:.1f}"
)

col5.metric(
    "Overtime %",
    f"{overtime_percentage:.2f}%"
)

# =========================================================
# ATTRITION DISTRIBUTION
# =========================================================

st.header("📈 Attrition Distribution")

col1, col2 = st.columns(2)

with col1:

    fig1 = px.pie(
        filtered_df,
        names="Attrition",
        hole=0.4,
        title="Attrition Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col2:

    fig2 = px.histogram(
        filtered_df,
        x="Attrition",
        color="Attrition",
        title="Attrition Count"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# DEPARTMENT ANALYSIS
# =========================================================

st.header("🏢 Department-wise Attrition")

fig3 = px.histogram(
    filtered_df,
    x="Department",
    color="Attrition",
    barmode="group",
    title="Department Attrition Analysis"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =========================================================
# SALARY ANALYSIS
# =========================================================

st.header("💰 Salary Analysis")

fig4 = px.box(
    filtered_df,
    x="Attrition",
    y="MonthlyIncome",
    color="Attrition",
    title="Salary vs Attrition"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =========================================================
# OVERTIME ANALYSIS
# =========================================================

st.header("⏰ Overtime Impact")

fig5 = px.histogram(
    filtered_df,
    x="OverTime",
    color="Attrition",
    barmode="group",
    title="Overtime vs Attrition"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# =========================================================
# JOB SATISFACTION
# =========================================================

st.header("😊 Job Satisfaction Analysis")

fig6 = px.histogram(
    filtered_df,
    x="JobSatisfaction",
    color="Attrition",
    barmode="group",
    title="Job Satisfaction Impact"
)

st.plotly_chart(
    fig6,
    use_container_width=True
)

# =========================================================
# WORK LIFE BALANCE
# =========================================================

st.header("⚖️ Work-Life Balance Analysis")

fig7 = px.histogram(
    filtered_df,
    x="WorkLifeBalance",
    color="Attrition",
    barmode="group",
    title="Work-Life Balance Impact"
)

st.plotly_chart(
    fig7,
    use_container_width=True
)

# =========================================================
# AGE ANALYSIS
# =========================================================

st.header("👥 Age Analysis")

fig8 = px.histogram(
    filtered_df,
    x="Age",
    color="Attrition",
    nbins=20,
    title="Age Distribution by Attrition"
)

st.plotly_chart(
    fig8,
    use_container_width=True
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📉 Feature Importance")

try:

    importance_df = pd.read_csv(
        "Employee_Attrition_Project/data/feature_importance.csv"
    )

    fig9 = px.bar(
        importance_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Affecting Attrition"
    )

    st.plotly_chart(
        fig9,
        use_container_width=True
    )

except:

    st.info(
        "Feature Importance CSV not found."
    )

# =========================================================
# PREDICTION SECTION
# =========================================================

st.header("🔮 Attrition Risk Predictor")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    years_at_company = st.slider(
        "Years At Company",
        0,
        40,
        5
    )

with col2:

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    total_working_years = st.slider(
        "Total Working Years",
        0,
        40,
        10
    )

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Attrition Risk"):

    try:

        prediction_df = pd.DataFrame(
            [df.drop("Attrition", axis=1).mode().iloc[0]]
        )

        prediction_df["Age"] = age
        prediction_df["MonthlyIncome"] = monthly_income
        prediction_df["OverTime"] = overtime
        prediction_df["JobSatisfaction"] = job_satisfaction
        prediction_df["WorkLifeBalance"] = work_life_balance
        prediction_df["YearsAtCompany"] = years_at_company
        prediction_df["EnvironmentSatisfaction"] = environment_satisfaction
        prediction_df["TotalWorkingYears"] = total_working_years

        prediction = model.predict(
            prediction_df
        )[0]

        probability = model.predict_proba(
            prediction_df
        )[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠ High Attrition Risk ({probability*100:.2f}%)"
            )

        else:

            st.success(
                f"✅ Low Attrition Risk ({(1-probability)*100:.2f}%)"
            )

        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(
            float(probability)
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )

# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================

st.header("📌 Business Recommendations")

st.success("""
✅ Reduce excessive overtime

✅ Improve work-life balance programs

✅ Increase employee engagement activities

✅ Review compensation structure

✅ Create career growth opportunities

✅ Improve manager-employee communication

✅ Focus on high-risk employee groups

✅ Conduct regular satisfaction surveys
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    **Developed by Nandini R**
    
    Data Analyst Internship Project
    
    Employee Attrition Analysis Dashboard
    """
)