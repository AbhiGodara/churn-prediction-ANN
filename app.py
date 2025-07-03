import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5') 

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehotencoder.pkl','rb') as file:
    onehotencoder = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


##Streamlit app
st.title("Bank Customer Churn Prediction")
st.write("Enter the customer details to predict if they will leave the bank or not.")

## user inputs
geography=st.selectbox("Geography", onehotencoder.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.number_input("Age", min_value=18, max_value=100, value=30)
credit_score=st.number_input("credit_score", min_value=300, max_value=850, value=600)
balance=st.number_input("Balance")
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products', 1, 4, 1)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])
estimated_salary=st.number_input("Estimated Salary", min_value=0, max_value=1000000, value=50000)

##Prepare the input data
input_data = {
    # 'Geography': geography,
    "Gender" : label_encoder_gender.transform([[gender]])[0],
    'Age': age,
    'CreditScore': credit_score,
    'Balance': balance,
    'Tenure': tenure,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])

# Encode categorical variables
geo_encoded=onehotencoder.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehotencoder.get_feature_names_out(["Geography"]))
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

with open('column_order.pkl', 'rb') as f:
    column_order = pickle.load(f)
input_df = input_df[column_order]

# Scale the input data
input_df_scaled=scaler.transform(input_df)
prediction=model.predict(input_df_scaled)
prediction_probability=prediction[0][0]

if prediction_probability >= 0.5:
    st.write("The Customer is likely to leave the bank with a probability of {:.2f}%".format(prediction_probability * 100))
else:
    st.write("The Customer is likely to stay with the bank with a probability of {:.2f}%".format((1 - prediction_probability) * 100))