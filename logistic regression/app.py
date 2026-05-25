import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
current_dir = os.path.dirname(__file__)

model_path = os.path.join(
    current_dir,
    "titanic_model.pkl"
)

model = joblib.load(model_path)
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("Titanic Survival Prediction App")
df = sns.load_dataset("titanic")
df = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]
]
df["age"].fillna(
    df["age"].median(),
    inplace=True
)

df["embarked"].fillna(
    df["embarked"].mode()[0],
    inplace=True
)

df = pd.get_dummies(
    df,
    columns=["sex", "embarked"],
    drop_first=True
)
st.subheader("Dataset")

st.dataframe(df.head())
X = df.drop("survived", axis=1)

y = df["survived"]
predictions = model.predict(X)
accuracy = accuracy_score(
    y,
    predictions
)

st.subheader("Accuracy")

st.write(f"Accuracy: {accuracy:.4f}")
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, predictions)

fig1, ax1 = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax1
)

ax1.set_title("Confusion Matrix")

st.pyplot(fig1)
st.subheader("Classification Report")

report = classification_report(
    y,
    predictions,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose()
)
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

st.dataframe(importance_df)

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.bar(
    importance_df["Feature"],
    importance_df["Coefficient"]
)

ax2.set_title("Feature Importance")

plt.xticks(rotation=45)

st.pyplot(fig2)
st.subheader("Manual Prediction")

pclass = st.selectbox(
    "Passenger Class",
    [1,2,3]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

sibsp = st.number_input(
    "Siblings/Spouses",
    min_value=0,
    value=0
)

parch = st.number_input(
    "Parents/Children",
    min_value=0,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

embarked_q = st.selectbox(
    "Embarked Q",
    [0,1]
)

embarked_s = st.selectbox(
    "Embarked S",
    [0,1]
)
sex_male = 1 if gender == "Male" else 0

input_data = pd.DataFrame({
    "pclass":[pclass],
    "age":[age],
    "sibsp":[sibsp],
    "parch":[parch],
    "fare":[fare],
    "sex_male":[sex_male],
    "embarked_Q":[embarked_q],
    "embarked_S":[embarked_s]
})
if st.button("Predict Survival"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    if prediction == 1:

        st.success(
            "Passenger Survived ✅"
        )

    else:

        st.error(
            "Passenger Did Not Survive ❌"
        )

    st.info(
        f"Survival Probability: {probability:.2f}"
    )