from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the path to your model directory
model_path = "./new"  # This is where your model files are located

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define sentiment labels
labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# FastAPI app instance
app = FastAPI()

class NewsContent(BaseModel):
    news_content: str

@app.post("/classify")
async def classify_news_content(news: NewsContent):
    """
    Classify the sentiment of the input news content sent via POST request.
    """
    try:
        # Preprocess and tokenize the input text
        inputs = tokenizer(news.news_content, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Perform inference
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()

        # Map the prediction to its corresponding label
        sentiment = labels.get(predictions, "Unknown")
        return {"sentiment": sentiment}
    except Exception as e:
        print(f"Error during classification: {e}")
        return {"error": "Error during classification."}
