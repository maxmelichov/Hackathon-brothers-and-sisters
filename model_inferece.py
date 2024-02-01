from transformers import pipeline
model_name = "distilbert-base-cased-finetuned"
classifier = pipeline('text-classification', model=model_name)

def predict(text: str) -> str:
    prediction = classifier(text)
    return prediction[0]["label"]
