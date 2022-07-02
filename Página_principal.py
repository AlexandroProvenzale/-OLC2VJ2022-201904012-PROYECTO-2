import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# funcion para clasificar las plantas
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'


def main():
    # Page config
    st.set_page_config(
        page_title="Proyecto 2",
        page_icon=":shark:",
    )

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

    # titulo
    st.title('Proyecto 2 - Machine learning')
    st.header('Alexandro Provenzale Pérez - 201904012')
    st.markdown("---")


if __name__ == '__main__':
    main()
