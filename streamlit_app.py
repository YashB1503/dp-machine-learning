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
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')


##Data Preperation
with st.sidebar:
  st.header('Input Feature')
  #"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.5)
  flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body Mass (gm)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))

  # Create a dataframe for the input feature
  data = {
    'island' : island,
    'bill_length_mm' : bill_length_mm,
    'bill_depth_mm' : bill_depth_mm,
    'flipper_length_mm' : flipper_length_mm,
    'body_mass_g' : body_mass_g,
    'gender' : gender
  }

  input_df = pd.DataFrame(data, index = 0)
  input_df
