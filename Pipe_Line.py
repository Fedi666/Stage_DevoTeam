import pandas as pd
import spacy


nlp = spacy.load("en_core_sci_sm")

def extract_symptoms(text):
    
    if pd.isna(text) or not str(text).strip():
        return ""
    doc = nlp(str(text))
    entities = [ent.text for ent in doc.ents]
    return "; ".join(entities) if entities else ""

def process_symptom_file(input_path, output_path):
    
   
    df = pd.read_csv(input_path)

  
    df['extracted_symptoms'] = df['Symptoms'].apply(extract_symptoms)

  
    df.to_csv(output_path, index=False)
    print(f"Done: {output_path}")

input_file = "C:\\Users\\fedib\\OneDrive\\Desktop\\Stage\\Diseases_Symptoms.csv"
output_file = "C:\\Users\\fedib\\OneDrive\\Desktop\\Stage\\Clean.csv"

process_symptom_file(input_file, output_file)
