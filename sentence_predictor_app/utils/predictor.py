import pandas as pd
import numpy as np
import joblib

def predict_sentence_length(age, race, offense_group, model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    age_scaled = scaler.transform(np.array([[age]]))[0][0]

    input_data = {'Age': age_scaled}
    for feature in model.feature_names_in_:
        if feature.startswith('Race_'):
            input_data[feature] = 1 if feature == f'Race_{race}' else 0
        elif feature.startswith('Offense_Group_'):
            input_data[feature] = 1 if feature == f'Offense_Group_{offense_group}' else 0

    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_).fillna(0)
    return model.predict(input_df)[0]
