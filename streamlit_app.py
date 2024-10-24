import streamlit as st
import pickle
import numpy as np

# Load the trained pipeline
@st.cache(allow_output_mutation=True)
def load_model():
    with open('xgb_pipeline.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Streamlit UI for input
st.title("Sales Forecasting App")

# Input fields for user to provide data
Item_MRP = st.number_input("Item Most Reselling Price (min value=0 , max value=500)", min_value=0.0, max_value=500.0, value=150.0)
Item_Visibility = st.number_input("Item Visibility (min_value=0 , max_value=0.2)", min_value=0.0, max_value=0.2, value=0.05)
Item_Weight = st.number_input("Item Weight ( min value>0, max value=20)", min_value=0.0, max_value=20.0, value=10.0)
Outlet_Establishment_Year = st.number_input("Outlet Establishment Year (min value=1980 , max value=2020)", min_value=1980, max_value=2020, value=1999)
Outlet_Type_Supermarket_Type1 = st.number_input("is Outlet Type1 (value= 0 or 1)", min_value=0, max_value=1, value=1)
Outlet_Type_Supermarket_Type2 = st.number_input("is Outlet Type2 (value= 0 or 1) ", min_value=0, max_value=1, value=0)
Outlet_Type_Supermarket_Type3 = st.number_input("is Outlet Type3 (value= 0 or 1) ", min_value=0, max_value=1, value=0)
Outlet_Identifier_OUT018 = st.number_input("is Outlet Identifier OUT018 (value= 0 or 1)", min_value=0, max_value=1, value=0)
Outlet_Identifier_OUT027 = st.number_input("is Outlet Identifier OUT027 (value= 0 or 1) ", min_value=0, max_value=1, value=0)


# Create input array for prediction
input_data = np.array([[Item_MRP, Item_Visibility, Outlet_Type_Supermarket_Type1, Item_Weight,
                        Outlet_Type_Supermarket_Type3, Outlet_Identifier_OUT027, Outlet_Establishment_Year,
                        Outlet_Identifier_OUT018, Outlet_Type_Supermarket_Type2]])

# When button is clicked, predict the sales
if st.button('Predict Sales'):
    prediction = model.predict(input_data)
    st.write(f"Predicted Sales: {prediction[0]:.2f}")
