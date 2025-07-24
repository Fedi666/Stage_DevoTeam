This project is a system designed to assist in predicting possible diseases based on user-described symptoms. The model leverages a combination of Natural Language Processing (NLP) and Machine Learning (ML) techniques to deliver structured, meaningful predictions—even from free-form symptom descriptions written in natural language.

It is built for educational purposes, and can serve as a prototype for future clinical decision-support systems.
----------------------------------------------------------------------------------------------------------------------
Features:

* Flexible Symptom Input: Users can enter full sentences or unstructured symptom descriptions.

* Symptom Extraction: The system uses an NLP model (LLM or SpaCy) to extract structured symptoms from natural text.

* Dataset Augmentation: A script augments underrepresented disease classes by generating variations of symptom sets.

* Symptom Encoding: Extracted symptoms are encoded using a multi-label binarization technique to match the model's input format.

* Disease Prediction: A trained Random Forest Classifier predicts the most likely disease based on encoded symptoms.

* Evaluation: Reports macro and weighted F1-scores to assess multi-class performance.

* Model Persistence: Trained models and encoders are saved using joblib for future use.

* Stack & Technologies
Python 3.10+

pandas, scikit-learn, joblib, spacy (for LLM-based extraction)

RandomForest For Classification

Tkinter for GUI


-----------------------------------------------------------------------------------------------
 Structure

- FinalPipeline.py               # Model training script
- Enrish3.py                     # Data augmentation logic
- Gui.py                         # GUI interface for predictions
- README.md                      # Project documentation 
- Pipe_Line.py                   #Symptom extraction and unifiying script
- Randomforestsymtomtest.py      #Prediction without GUi (if needed)  


-----------------------------------------------------------------------------------------------
*How to use:

1) Run the Pipe_Line Script to get a clean data set with extracted symptoms : input=Original Dataset //output=clean.csv
2) Run Enrish script TO augment the clean data set (it is required to have a large dataset for the model to work properly) : input=clean.csv // output=Enrish.csv
3) Run Final Pipeline to Get the exact coding of symptoms needed for the model to work + Random Forest Model it self : input Enrish.csv // output=Model+Codage
4) Run Gui Script to get an interface to test the results
**optional run Randomforestsymtomtest to test without a GUI   


------------------------------------------------------------------------------------------------
*Notes:


- Disease codes that appear only once are automatically excluded to avoid bias during training.

- The system currently works best for diseases with multiple entries in the dataset.

- Correct Spelling of symptoms is advised

