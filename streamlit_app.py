import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw
  
  st.write('**Y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')


##Input Feature

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
    'sex' : gender
  }

  input_df = pd.DataFrame(data, index = [0])
  input_pengiuns = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Feature'):
  st.write('**X**')
  input_df
  st.write('**Combined Data**')
  input_pengiuns

# Data Preperation
#Encode X
encode = ['island', 'sex']
df_pengiuns = pd.get_dummies(input_pengiuns, prefix=encode)
X = df_pengiuns[1:]
input_row = df_pengiuns[:1]

#Encode Y  
target_mapper = {
    'Adelie' : 0,
    'Chinstrap' : 1,
    'Gentoo' : 2
  }

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preperation'):
  st.write('**Encoded input pengiuns [X]**')
  input_row
  st.write('**Encoded y**')
  y

##Model training and inference
clf = RandomForestClassifier()
clf.fit(X, y)

## apply the model to make predection
prediction = clf.predict(input_row)
predection_proba = clf.predict_proba(input_row)

predection_proba
