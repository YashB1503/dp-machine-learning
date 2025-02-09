import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X
  
  st.write('**Y**')
  Y = df.species
  Y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_gm', color='species')
