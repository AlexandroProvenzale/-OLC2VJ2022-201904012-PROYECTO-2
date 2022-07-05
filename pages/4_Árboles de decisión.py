import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn import preprocessing

st.title("Árboles de decisión")
data = st.file_uploader("Cargar archivo", type=["csv", "json", "xls", "xlsx"], accept_multiple_files=False)

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
    select_columnas = st.multiselect("Seleccione las columnas de las variables de entrada", all_columns,
                                     default=all_columns)
    select_salida = st.selectbox("Selecciones columna de la variable de salida", all_columns)
    le = preprocessing.LabelEncoder()

    st.header("Previsualización")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Variables de entrada")
        new_df = df[select_columnas]
        st.dataframe(new_df)
    with col2:
        st.subheader("Variable de salida")
        st.dataframe(df[select_salida])

    listaLE = []
    listaEquivalente = []
    for data in select_columnas:
        actual_encoded = le.fit_transform(np.asarray(df[data]))
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        listaLE.append(actual_encoded)
        listaEquivalente.append(le_name_mapping)
    salida = le.fit_transform(np.asarray(df[select_salida]))
    listaSalida = dict(zip(le.classes_, le.transform(le.classes_)))

    features = list(zip(*listaLE))

    fig, ax = plt.subplots()
    clf = DecisionTreeClassifier().fit(features, salida)
    plot_tree(clf, filled=True)
    st.pyplot(fig)
else:
    st.warning("Aún no se ha cargado un archivo")
