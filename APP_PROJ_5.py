#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[3]:





# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# Read the data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\ASUS\Desktop\USC\24SP\DSO_522\Project05\Joined_Data.csv", index_col=False)
    return data

# Preprocessing function
def preprocess_data(data):
    # Handle missing values
    data.dropna(inplace=True)

    # Convert 'onpromotion' column to binary
    data['onpromotion'] = data['onpromotion'].apply(lambda x: 1 if x > 0 else 0).astype('int8')

    # Create 'Dept_Daily_Sales_Agg' column
    data['Dept_Daily_Sales_Agg'] = data.groupby(['store_nbr', 'date'])['sales'].transform('sum')

    # Convert 'type_y' column to numerical
    #data['type_y'].replace(['A', 'B', 'C', 'D', 'E'], [1, 2, 3, 4, 5], inplace=True)

    # Pivot table
    data = data.pivot_table(values=['onpromotion', 'dcoilwtico', 'holiday', 'sales', 'cluster'], 
                            index=['date', 'store_nbr'], 
                            aggfunc={'onpromotion': "sum", 
                                     'dcoilwtico': 'mean', 
                                     'holiday': "mean", 
                                     'sales': 'sum', 
                                     'cluster': 'mean'}).reset_index()

    # Create 'High Sales Volume' column
    median_sales = data.sales.median()
    data['High Sales Volume'] = data['sales'].apply(lambda x: '1' if x >= median_sales else '0')

    # Sort the data
    data = data.sort_values(by=['store_nbr', 'date'])

    # Shift sales columns
    data['-1_day_sales'] = data.groupby('store_nbr')['sales'].shift(1)
    data['-2_day_sales'] = data.groupby('store_nbr')['sales'].shift(2)
    data.dropna(inplace=True)

    # Scale numerical features
    numeric_cols = ['dcoilwtico', 'onpromotion', 'sales', '-1_day_sales', '-2_day_sales']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

# Neural network model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(6, activation=tf.nn.relu),
        keras.layers.Dense(2, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    st.title("Sales Prediction")

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Data loaded successfully!')

    # Preprocess the data
    st.subheader('Preprocessed Data')
    processed_data = preprocess_data(data)
    st.write(processed_data.head())

    # Build the neural network model
    st.subheader('Neural Network Model')
    model = build_model()

    # Train the model
    st.text('Training the neural network model...')
    X_train = processed_data[['dcoilwtico', 'onpromotion', 'sales', '-1_day_sales', '-2_day_sales']]
    y_train = processed_data['High Sales Volume'].astype(int)
    model.fit(X_train, y_train, epochs=10, verbose=0)
    st.text('Neural network model trained successfully!')

    # Show model evaluation
    st.subheader('Model Evaluation')
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    st.write(f'Training Accuracy: {train_acc:.2f}')
    st.write(f'Training Loss: {train_loss:.2f}')

if __name__ == '__main__':
    main()


