import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("Mental Disorder Prediction App")

# -----------------------------
# Load and preprocess data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset-Mental-Disorders.csv")
    df = df.drop("Patient Number", axis=1)

    ipcols = df.columns.drop("Expert Diagnose")

    df_encoded = pd.get_dummies(df[ipcols])
    df_final = pd.concat([df["Expert Diagnose"], df_encoded], axis=1)

    X = df_final.drop("Expert Diagnose", axis=1)
    y = df_final["Expert Diagnose"]

    return X, y


# -----------------------------
# Train model (cached)
# -----------------------------
@st.cache_resource
def train_model():
    X, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    return model, accuracy, X.columns


# -----------------------------
# Main App
# -----------------------------
model, accuracy, feature_columns = train_model()

st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

st.header("Make a Prediction")

# Simple demo input (all zeros by default)
input_data = {}

for col in feature_columns[:10]:  # limit to first 10 to keep it light
    input_data[col] = st.number_input(f"{col}", min_value=0, max_value=1, value=0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"Predicted Mental Disorder: {prediction[0]}")
