import joblib


clf = joblib.load("random_forest_model_20250723_180514.pkl")  
mlb = joblib.load("symptom_encoder_20250723_180514.pkl")


input_symptoms = ["High blood sugar levels ","pregnancy"]


symptom_input = mlb.transform([input_symptoms])


predicted_code = clf.predict(symptom_input)
print("Predicted Disease Code:", predicted_code[0])
