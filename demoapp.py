import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load fine-tuned sentiment analysis model
model_path = "./fine_tuned_model"  # Path to the saved fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(model_path)  # Load the model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)  # Load the tokenizer


# Define a function to predict sentiment from text
def predict_sentiment(text):
    # Tokenize the input text and convert to PyTorch tensor
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

    # Make a prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Determine the predicted sentiment class and convert to a sentiment label
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "positive" if predicted_class == 1 else "negative"

    return sentiment


# Create a Gradio interface for sentiment prediction
iface = gr.Interface(
    fn = predict_sentiment,
    inputs = "text",
    outputs = "text",
    title = "Sentiment Analysis Of IMDb Reviews",
    description = "Enter a text and get sentiment prediction.",
)

# Launching the Gradio interface
iface.launch()
