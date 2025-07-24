import tkinter as tk
from tkinter import messagebox
import joblib
import spacy


nlp = spacy.load("en_core_sci_sm")


clf = joblib.load("Model.pkl")
mlb = joblib.load("Codage.pkl")


def extract_symptoms(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]


def predict_disease():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter a symptom description.")
        return

    symptoms = extract_symptoms(user_input)
    if not symptoms:
        result_var.set("No recognizable symptoms found.")
        return

    try:
        encoded = mlb.transform([symptoms])
        prediction = clf.predict(encoded)
        result_var.set(f"Predicted Disease Code: {prediction[0]}")
    except Exception as e:
        result_var.set(f"Prediction Error: {e}")


root = tk.Tk()
root.title("Disease Predictor")
root.geometry("500x300")

title_label = tk.Label(root, text="Symptom-Based Disease Predictor", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

entry_label = tk.Label(root, text="Describe your symptoms below:")
entry_label.pack()

entry = tk.Text(root, height=5, width=60)
entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict Disease Code", command=predict_disease)
predict_button.pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()
