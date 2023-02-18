import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()
plt.style.use('seaborn-whitegrid')
from streamlit.web.cli import main
import streamlit as st
st.title("Future Profit Prediction Model")
df = st.text_input("Let's Predict the Future Profit")



df = pd.read_csv("Profit Prediction using Linear Regression.csv")
print(df.head())
X = df[["Marketing Spend", "Administration", "Transport"]]
y = df["Profit"]
X = X.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df = pd.DataFrame(data= {"Predicted Profit": y_pred.flatten()})
df.head()
st.write(y_pred)