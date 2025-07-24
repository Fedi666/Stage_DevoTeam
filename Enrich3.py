import pandas as pd
import random


df = pd.read_csv("Clean.csv")


def generate_symptom_variant(symptom_str):
    symptoms = [s.strip() for s in symptom_str.split(';') if s.strip()]
    if len(symptoms) < 2:
        return symptom_str

    base = list(set(symptoms))

    variation = random.sample(base, max(1, len(base) - random.randint(0, 2)))
    if random.random() > 0.5:
        variation.append(random.choice(base) + " related")

    return '; '.join(variation)


df_valid = df[df['extracted_symptoms'].notna()]

augmented_rows = []


for _, row in df_valid.iterrows():
    for _ in range(50):
        new_row = row.copy()
        new_row['extracted_symptoms'] = generate_symptom_variant(row['extracted_symptoms'])
        augmented_rows.append(new_row)


augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)


augmented_df.to_csv("Enrich3.csv", index=False)

print("saved as 'Enrich3.csv'")
