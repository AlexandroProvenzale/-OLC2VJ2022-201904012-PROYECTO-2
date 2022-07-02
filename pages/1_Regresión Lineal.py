import streamlit as st
import pandas as pd
import os

st.title("Regresión lineal")
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

# titulo de sidebar
st.sidebar.header('User Input Parameters')


# funcion para poner los parametrso en el sidebar
def user_input_parameters():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    datas = {'sepal_length': sepal_length,
             'sepal_width': sepal_width,
             'petal_length': petal_length,
             'petal_width': petal_width,
             }
    features = pd.DataFrame(datas, index=[0])
    return features


df = user_input_parameters()

# escoger el modelo preferido
option = ['Linear Regression', 'Logistic Regression', 'SVM']
model = st.sidebar.selectbox('Which model you like to use?', option)

st.subheader(model)
st.write(df)
