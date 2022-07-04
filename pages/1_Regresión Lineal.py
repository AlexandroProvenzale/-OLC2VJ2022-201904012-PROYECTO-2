import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

st.title("Regresión lineal")
data = st.file_uploader("Cargar archivo", type=["csv", "json", "xls", "xlsx"], accept_multiple_files=False)

# titulo de sidebar
st.sidebar.header('Selección de operación')
opciones = ['Graficar puntos', 'Predicción de la tendencia']
model = st.sidebar.selectbox('¿Qué operación desea realizar?', opciones)

if data is not None:
    st.success("Archivo cargado exitosamente")
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

    x = np.asarray(df[select_columnX]).reshape(-1, 1)
    y = df[select_columnY]

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)

    if model == "Predicción de la tendencia":
        select_predict = st.number_input("Ingrese valor a predecir", step=1)
        st.subheader("Predicción:")
        if select_predict == 0:
            st.warning("Aún no ingresa un dato para hacer predicción")
        else:
            st.subheader(str(regr.predict([[select_predict]])[0]))
    elif model == "Graficar puntos":
        fig, ax = plt.subplots()
        plt.title("Regresión lineal - " + split_t[0])
        plt.xlabel(select_columnX)
        plt.scatter(x, y, color='black')
        plt.plot(x, y_pred, color='blue', linewidth=2)
        st.pyplot(fig)
        if st.checkbox("Mostrar función"):
            # st.write(regr.coef_)
            st.latex(r"f(x) = " + "(" + str(regr.coef_[0]) + ")" + "x + " + "(" + str(regr.intercept_) + ")")
else:
    st.warning("Aún no se ha cargado un archivo")
