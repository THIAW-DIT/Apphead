# import libraries
import pandas as pd
# import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import streamlit as st

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    data = pd.read_csv(file_path)
    return data
file_path = "diabetes.csv" 
data_d = load_data(file_path)
print(data_d.head(5))
print("----------------------------------------------------------------------------------------------")
# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using appropriate imputation
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    # data = data.isnull().sum()
    # print(data)
    
    # Deal with outlier data 
    for column in data_d.columns:
        q1 = np.percentile(data_d[column], 25)
        q3 = np.percentile(data_d[column], 75) 
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        data_d.loc[(data_d[column] < lower_bound), column] = lower_bound
        data_d.loc[(data_d[column] > upper_bound), column] = upper_bound
    return data_d
data_p = preprocess_data(data_d)
data_p.head(5)

# Verifier le traitement des valeurs aberantes
plt.figure(figsize = (14,7))
sns.boxplot(data = data_p)
plt.grid()
plt.show() 
    
Y = data_p['Outcome']
X = data_p.drop(['Outcome'], axis = 1)
sc = StandardScaler()
X_cr = sc.fit_transform(X)

# Function to split data into training and testing sets (5 pts)
def split_data(data):
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_cr, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = split_data(data_p)

    

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(X_train, y_train)
    # Return best model
    return model
    

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the best model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
# Print the mesure of precision
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    
    # Deploy the best model using Streamlit or Flask (bonus)
    st.title("Application de prediction du diabetes")
    st.sidebar.header(" Donnees d'entrees de l'utilisateur")

    # Create input fields for user to provide data
    # Replace the following with the actual feature names in your dataset
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17)
    glucose = st.sidebar.slider("Glucose", 0, 199)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99)
    insulin = st.sidebar.slider("Insulin", 0, 846)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42)
    age = st.sidebar.slider("Age", 21, 81)

    # Create a DataFrame with the user input
    user_input = pd.DataFrame(
        data=[[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    )

    # Use the model to make predictions
    prediction = model.predict(user_input)

    # Display the prediction result
    st.subheader("Resultat de la prediction")
    if prediction[0] == 0:
        st.write(" **No Diabetes** ")
    else:
        st.write(" **Diabetes** ")

# Main function
def main():
    # Load data
    data = load_data(file_path)
    
    # Preprocess data
    data_processed = preprocess_data(data)
    # Split data
    X_train, X_test, y_train, y_test = split_data(data_processed)
    # Train a model with hyperparameters
    model = train_model(X_train, y_train)
    # Evaluate the model
    evaluate_model(model, X_test, y_test) 
    # Deploy the model (bonus)
    deploy_model(model, X_test)

if __name__ == "__main__":
    main()











