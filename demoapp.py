import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Loading the checkpoint model from fine-tuning
checkpoint_path = "./sentiment_model/checkpoint-1500"  # Update this path accordingly
checkpoint_model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)

# Defining a function to make predictions using the checkpoint model
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = checkpoint_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(),
    title="IMDb Reviews Sentiment Analysis",
    description="Enter a review and get sentiment prediction using the model.",
)

# Launch the Gradio interface
iface.launch()
