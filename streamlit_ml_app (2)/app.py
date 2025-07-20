
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("Iris Flower Prediction App 🌸")
st.write("Enter flower measurements to predict the species.")

# User input
sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.0)
sepal_width  = st.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 4.0)
petal_width  = st.slider('Petal width (cm)', 0.1, 2.5, 1.0)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
pred_proba = model.predict_proba(input_data)

species = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction:")
st.write(f"🌺 **{species[prediction[0]]}**")

# Probability Chart
st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame(pred_proba, columns=species)
st.bar_chart(proba_df.T)

# Visual explanation
if st.checkbox("Show sample distribution"):
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, ax=ax)
    st.pyplot(fig)
