# Import required libraries
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize and preprocess the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization function to the dataset in batches
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir="./sentiment_model",         # Directory to save model checkpoints and logs
    evaluation_strategy="steps",            # Evaluate during training at certain steps
    save_strategy="steps",                  # Save model during training at certain steps
    learning_rate=1e-5,                    
    per_device_train_batch_size=8,          # Batch size for training
    per_device_eval_batch_size=8,           # Batch size for evaluation
    num_train_epochs=1,                     # Number of training epochs (1 epoch here) to lower the training time
    eval_steps=500,                         # Evaluate every 500 steps
    save_steps=500,                         # Save model every 500 steps
)

# Create a Trainer instance for training the model
trainer = Trainer(
    model=model,                           
    args=training_args,                    
    train_dataset=tokenized_datasets["train"],  # Training dataset after tokenization
    eval_dataset=tokenized_datasets["test"],    # Evaluation dataset after tokenization
)

# Train the model using the Trainer instance
trainer.train()

# Save the fine-tuned model after training
model.save_pretrained("./fine_tuned_model")
