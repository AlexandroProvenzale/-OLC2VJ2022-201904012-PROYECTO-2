import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing

st.title("Redes neuronales")
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

    mlp = MLPClassifier(hidden_layer_sizes=(len(features), len(features), len(features)), max_iter=300, alpha=0.0001,
                        solver='adam', random_state=21, tol=0.000000001)

    mlp.fit(features, salida)

    cols = st.columns(len(select_columnas))
    prediction_boxes = []
    for i, columna in enumerate(select_columnas):
        with cols[i]:
            prediction_boxes.append(st.selectbox(columna, listaEquivalente[i].keys()))
    st.subheader(prediction_boxes)
    if st.button("Evaluar"):
        predictoria = []
        for i, box in enumerate(prediction_boxes):
            predictoria.append(listaEquivalente[i].get(box))
        predicted = mlp.predict([[*predictoria]])
        st.subheader("Predicción:")
        st.subheader(list(listaSalida.keys())[list(listaSalida.values()).index(predicted)])
else:
    st.warning("Aún no se ha cargado un archivo")
