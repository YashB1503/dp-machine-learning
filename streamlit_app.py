import streamlit as st

st.title('🎈 Machine Learning App')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df

st.write('This is app builds a machine learning model!')
