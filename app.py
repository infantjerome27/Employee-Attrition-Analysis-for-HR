import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

st.title("Employee Attrition Analysis for HR")
st.caption("Predict turnover, identify key drivers, and monitor HR KPIs in one dashboard.")

@st.cache_data

def load_data():
    # Use a realistic HR analytics dataset with common attrition fields.
    df = pd.DataFrame(
        {
            "Age": np.random.randint(22, 60, 1200),
            "Salary": np.random.choice([30000, 45000, 60000, 80000, 110000], 1200, p=[0.25, 0.30, 0.20, 0.15, 0.10]),
            "Department": np.random.choice(["Sales", "HR", "Engineering", "Marketing", "Operations"], 1200, p=[0.25, 0.10, 0.30, 0.15, 0.20]),
            "YearsAtCompany": np.random.randint(0, 15, 1200),
            "Overtime": np.random.choice(["Yes", "No"], 1200, p=[0.35, 0.65]),
            "JobSatisfaction": np.random.randint(1, 5, 1200),
            "Attrition": np.random.choice(["Yes", "No"], 1200, p=[0.18, 0.82]),
        }
    )
    # Inject stronger signal to make the attrition pattern plausible.
    df.loc[df["YearsAtCompany"] < 2, "Attrition"] = np.where(np.random.rand(len(df[df["YearsAtCompany"] < 2])) < 0.45, "Yes", "No")
    df.loc[df["Salary"] < 40000, "Attrition"] = np.where(np.random.rand(len(df[df["Salary"] < 40000])) < 0.35, "Yes", "No")
    df.loc[df["Age"] > 50, "Attrition"] = np.where(np.random.rand(len(df[df["Age"] > 50])) < 0.25, "Yes", "No")
    return df


df = load_data()

st.subheader("HR dashboard KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", len(df))
col2.metric("Attrition Rate", f"{(df['Attrition'].eq('Yes').mean() * 100):.1f}%")
col3.metric("Average Age", f"{df['Age'].mean():.1f} yrs")
col4.metric("Average Salary", f"${df['Salary'].mean():,.0f}")

st.subheader("Attrition by department and salary band")
left, right = st.columns(2)
attrition_department = df.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean()).reset_index(name="Attrition Rate")
left.plotly_chart(px.bar(attrition_department, x="Department", y="Attrition Rate", color="Department", title="Attrition Rate by Department"), use_container_width=True)

salary_bins = pd.cut(df["Salary"], bins=[0, 40000, 65000, 90000, 150000], labels=["Low", "Mid", "High", "Executive"])
df_bin = df.copy()
df_bin["SalaryBand"] = salary_bins
salary_band = df_bin.groupby("SalaryBand")["Attrition"].apply(lambda s: (s == "Yes").mean()).reset_index(name="Attrition Rate")
right.plotly_chart(px.bar(salary_band, x="SalaryBand", y="Attrition Rate", title="Attrition Rate by Salary Band"), use_container_width=True)

st.subheader("Key turnover factors")
fig_age = px.box(df, x="Attrition", y="Age", color="Attrition", title="Attrition vs Age")
fig_salary = px.box(df, x="Attrition", y="Salary", color="Attrition", title="Attrition vs Salary")
fig_tenure = px.box(df, x="Attrition", y="YearsAtCompany", color="Attrition", title="Attrition vs Tenure")
st.plotly_chart(fig_age, use_container_width=True)
st.plotly_chart(fig_salary, use_container_width=True)
st.plotly_chart(fig_tenure, use_container_width=True)

st.subheader("Predictive model")
X = df[["Age", "Salary", "Department", "YearsAtCompany", "Overtime", "JobSatisfaction"]]
y = df["Attrition"].eq("Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

categorical = ["Department", "Overtime"]
numerical = ["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

model_choice = st.selectbox("Choose classification algorithm", ["Logistic Regression", "Random Forest"])
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=2000)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.write("Model accuracy:", round((y_pred == y_test).mean(), 3))
st.text("Classification report:\n" + classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
st.dataframe(cm_df)

st.subheader("What influences resignation most?")
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feature_importance = None
if model_choice == "Random Forest":
    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
else:
    coef = pipeline.named_steps['classifier'].coef_[0]
    feature_importance = pd.Series(coef, index=feature_names).abs().sort_values(ascending=False)

st.bar_chart(feature_importance.head(10))

st.subheader("HR action notes")
st.info("High turnover is often linked to lower salary, early tenure, and department-specific pressures. Use this dashboard to prioritize retention interviews, compensation reviews, and manager support.")
