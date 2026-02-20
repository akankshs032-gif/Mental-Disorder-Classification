import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout='wide')
st.title('Mental Disorder Prediction App')
st.write('This app predicts mental disorders based on various symptoms using a Decision Tree Classifier.')

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('/content/Dataset-Mental-Disorders.csv')

    # Drop 'Patient Number'
    df.drop('Patient Number', axis=1, inplace=True)

    # Identify input columns (all except 'Expert Diagnose')
    ipcols = df.columns.drop('Expert Diagnose')

    # One-hot encode input columns
    df1_list = []
    for col in ipcols:
        df1_list.append(pd.get_dummies(df[col], prefix=col))
    df1 = pd.concat(df1_list, axis=1).astype(int)

    # Drop original input columns and concatenate with encoded columns
    df_final = pd.concat([df.drop(ipcols, axis=1), df1], axis=1)

    # Separate features (ip) and target (op)
    ip = df_final.drop('Expert Diagnose', axis=1)
    op = df_final['Expert Diagnose']

    return ip, op

# Function to train the model
@st.cache_resource
def train_model(ip, op):
    x_train, x_test, y_train, y_test = train_test_split(ip, op, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # Evaluate the model
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)

    return model, accuracy, x_train.columns


# Main application logic
if __name__ == '__main__':
    st.header('1. Data Loading and Preprocessing')
    ip, op = load_and_preprocess_data()
    st.write(f"Original dataset shape: {ip.shape[0]} rows, {ip.shape[1]} features (after encoding)")
    st.dataframe(ip.head())

    st.header('2. Model Training and Evaluation')
    model, accuracy, feature_columns = train_model(ip, op)
    st.write(f'Decision Tree Classifier trained with an accuracy of: {accuracy:.2f}')

    st.header('3. Make New Predictions')
    st.write('Enter values for the features to get a prediction. (For demonstration, using the first row of processed data as an example input.)')

    # Example: Use the first row of the preprocessed data for a dummy prediction input
    # In a real app, you'd use st.sidebar or st.columns with individual st.selectbox for each feature
    example_input = ip.iloc[0].copy()

    # Display example input for manual modification in a real app
    st.subheader('Example Input Features (modify these to make a prediction)')
    edited_input = {} # Dictionary to hold user modified inputs

    # For simplicity, we'll just show the first few features as text
    # A full interactive input would require many selectboxes/sliders based on `feature_columns`
    display_cols = feature_columns[:10] # Show first 10 for brevity
    input_values = []
    for col in display_cols:
        val = st.text_input(f'Value for {col}', str(example_input[col]), key=col)
        edited_input[col] = int(val) if val.isdigit() else example_input[col] # Basic type conversion

    # Create a dummy input for prediction (using the example_input for now)
    # In a real app, `edited_input` would be converted to a DataFrame row.
    sample_for_prediction = pd.DataFrame([example_input], columns=feature_columns)

    if st.button('Predict'):
        prediction = model.predict(sample_for_prediction)
        st.success(f'The predicted mental disorder is: **{prediction[0]}**')
