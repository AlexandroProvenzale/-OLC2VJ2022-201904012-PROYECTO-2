import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model

st.title("Regresión polinomial")
data = st.file_uploader("Cargar archivo", type=["csv", "json", "xls", "xlsx"], accept_multiple_files=False)

# titulo de sidebar
st.sidebar.header('Selección de operación')
opciones = ['Graficar puntos', 'Predicción de la tendencia']
sidebar = st.sidebar.selectbox('¿Qué operación desea realizar?', opciones)

st.markdown("""
    <style>
    .css-1lsmgbg.egzxvld0 {
        visibility: hidden;
    }
    .css-14xtw13.e8zbici0 {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

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

    polynomial_features = PolynomialFeatures(degree=int(select_degree))
    X_TRANSF = polynomial_features.fit_transform(x)

    model = linear_model.LinearRegression()
    model.fit(X_TRANSF, y)

    Y_NEW = model.predict(X_TRANSF)

    if sidebar == "Predicción de la tendencia":
        # Error cuadrático medio (MSE)
        mse = mean_squared_error(y, Y_NEW)
        # Raíz del error cuadrático (RMSE)
        rmse = np.sqrt(mean_squared_error(y, Y_NEW))
        # Coeficiente R^2
        r2 = r2_score(y, Y_NEW)

        # Prediction
        x_new_min = x.min()
        x_new_max = st.number_input("Ingrese valor a predecir", step=1)

        X_NEW = np.linspace(x_new_min, x_new_max, 50)
        X_NEW = X_NEW[:, np.newaxis]
        X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

        st.subheader("Predicción:")
        if x_new_max == 0:
            st.warning("Aún no ingresa un dato para hacer predicción")
        else:
            temp = model.predict(X_NEW_TRANSF)
            st.subheader(str(temp[(len(temp) - 1)]))
    elif sidebar == "Graficar puntos":
        x_new_min = x.min()
        x_new_max = x.max()

        X_NEW = np.linspace(x_new_min, x_new_max, 50)
        X_NEW = X_NEW[:, np.newaxis]
        X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

        Y_NEW = model.predict(X_NEW_TRANSF)
        fig, ax = plt.subplots()
        plt.title("Regresión polinomial - " + split_t[0])
        plt.plot(X_NEW, Y_NEW, color='coral', linewidth=2)
        plt.grid()
        plt.xlim(x_new_min, x_new_max)
        plt.xlabel(select_columnX)
        st.pyplot(fig)
        if st.checkbox("Mostrar función"):
            coefs = model.coef_
            co_latex = "f(x) = "
            i = int(select_degree)
            intercept = 0
            for inter in model.intercept_:
                intercept += inter
            for cof in model.coef_[0, ::-1]:
                if i > 0:
                    co_latex += "(" + str(cof) + ")" + "x^" + str(i) + " + "
                    i -= 1
                else:
                    co_latex += "(" + str(intercept) + ")"
                    break
            # st.write(regr.coef_)
            st.latex(co_latex)

else:
    st.warning("Aún no se ha cargado un archivo")
