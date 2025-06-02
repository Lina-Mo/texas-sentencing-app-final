import streamlit as st
from utils.predictor import predict_sentence_length

st.title("🔍 Sentence Length Estimator")

age = st.slider("Select Age", 18, 80, 30)
race = st.selectbox("Select Race", ['Black', 'White', 'Hispanic', 'Other and Unknown'])
offense_group = st.selectbox("Select Offense Group", ['Drug', 'Violent', 'Sexual', 'Property', 'Public Order', 'Other'])

prediction = predict_sentence_length(age, race, offense_group,
                                     model_path='/mount/src/texas-sentencing-app-final/sentence_predictor_app/models/sentence_model.pkl',
                                     scaler_path='/mount/src/texas-sentencing-app-final/sentence_predictor_app/models/age_scaler.pkl')

st.subheader(f"🧾 Estimated Sentence Length: **{prediction:.2f} years**")
