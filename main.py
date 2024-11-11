from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Initialize FastAPI app
app = FastAPI()

# Define data model for request
class Message(BaseModel):
    text: str

# Initialize model components
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer()
model = XGBClassifier()

# Load and preprocess the dataset
spam_df = pd.read_csv('datasets/spam.csv', encoding='latin-1').rename(columns={'v1': 'LABEL', 'v2': 'TEXT'})[['LABEL', 'TEXT']]
dataset_df = pd.read_csv('datasets/Dataset_5971.csv')[['LABEL', 'TEXT']]
combined_df = pd.concat([spam_df, dataset_df], ignore_index=True)

# Change labels to "Phishing" and "Legitimate"
combined_df['LABEL'] = combined_df['LABEL'].map({'ham': 'Legitimate', 'spam': 'Phishing'})

# Encode labels and fit vectorizer
label_encoder.fit(combined_df['LABEL'])
X = vectorizer.fit_transform(combined_df['TEXT'])
y = label_encoder.transform(combined_df['LABEL'])

# Train the model
model.fit(X, y)

# Home endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the SMS Phishing Detection API!"}

# Prediction endpoint
@app.post("/predict")
def predict(message: Message):
    try:
        # Vectorize input text
        text_vector = vectorizer.transform([message.text])
        # Predict label
        prediction = model.predict(text_vector)
        # Decode label
        label = label_encoder.inverse_transform(prediction)[0]
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
