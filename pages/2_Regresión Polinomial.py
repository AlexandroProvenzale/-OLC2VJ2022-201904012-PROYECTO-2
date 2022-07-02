import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model

st.title("Regresión lineal")
data = st.file_uploader("Cargar archivo", type=["csv", "json", "xls", "xlsx"], accept_multiple_files=False)

# titulo de sidebar
st.sidebar.header('Selección de operación')
opciones = ['Graficar puntos', 'Predicción de la tendencia']
model = st.sidebar.selectbox('¿Qué operación desea realizar?', opciones)

if data is not None:
    split_t = os.path.splitext(data.name)
    extension = split_t[1]
    if extension == ".csv":
        df = pd.read_csv(data)
        st.dataframe(df.head())
    elif extension == ".json":
        df = pd.read_json(data)
        st.dataframe(df.head())
    elif extension == ".xls" or extension == ".xlsx":
        bytes_data = data.getvalue()
        df = pd.read_excel(bytes_data)
        st.dataframe(df.head())

    all_columns = df.columns.to_list()
    select_columnX = st.selectbox("Seleccione X", all_columns)
    select_columnY = st.selectbox("Seleccione Y", all_columns)
    select_degree = st.number_input("Ingrese grado polinomial", step=1)

    x = np.asarray(df[select_columnX])
    y = np.asarray(df[select_columnY])

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=select_degree)
    X_TRANSF = polynomial_features.fit_transform(x)

    model = linear_model.LinearRegression()
    model.fit(X_TRANSF, y)

    Y_NEW = model.predict(X_TRANSF)
    mse = mean_squared_error(y, Y_NEW)
    rmse = np.sqrt(mean_squared_error(y, Y_NEW))
    r2 = r2_score(y, Y_NEW)
