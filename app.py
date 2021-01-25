# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 01:50:25 2020
@author: jhon
"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):

    predictions_data = predict_model(estimator=model, data=df)

    return predictions_data['Label'][0]


model = load_model('Random_Forest')


st.title('Prediccion de abandono de empresa ')
st.write('Esta es una aplicación web para determinar la permanencia de colaboradores según \
         varias caracteristicas que puede ver en la barra lateral. Ajuste el \
         valor de cada colaborador. Después de eso, haga clic en el botón Predecir en la parte inferior para \
         ver la predicción del clasificador.')


satisfaction_level = st.sidebar.slider(label='Nivel de satisfaccion', min_value=0.1,
                                       max_value=1.0,
                                       value=0.1,
                                       step=0.1)

last_evaluation = st.sidebar.slider(label='Ultima evaluacion', min_value=0.1,
                                    max_value=1.0,
                                    value=0.1,
                                    step=0.1)

number_project = st.sidebar.slider(label='Numero de proyectos', min_value=1,
                                   max_value=10,
                                   value=1,
                                   step=1)

time_spend_company = st.sidebar.slider(label='Años', min_value=1,
                                         max_value=10,
                                         value=1,
                                         step=1)

work_accident = st.sidebar.slider(label='Accidentes en el trabajo', min_value=0,
                                  max_value=10,
                                  value=1,
                                  step=1)

promotion_last_5years = st.sidebar.slider(label='Ascensos', min_value=0,
                                          max_value=10,
                                          value=1,
                                          step=1)

departments = st.sidebar.selectbox('area', ['sales', 'accounting', 'hr',
                                             'technical', 'management', 'product_mng', 'marketing', 'RandD'])
salary = st.sidebar.selectbox("Salario", ("low", "medium", "high"))


features = {'satisfaction_level': satisfaction_level, 'last_evaluation': last_evaluation,
            'number_project': number_project,'time_spend_company': time_spend_company, 
            'work_accident': work_accident,'promotion_last_5years': promotion_last_5years, 
            'departments': departments,'salary': salary
            }


features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predicción'):

    prediction = predict_quality(model, features_df)

    st.write(
        'Según los valores de las características, El colaborador : ' + str(prediction))
