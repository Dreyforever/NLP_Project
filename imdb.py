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

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments for fine-tuning with checkpoints
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,    # Increase number of epochs for more checkpoints
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,    # Limit the number of checkpoints to save
    load_best_model_at_end=True,  # Load the best model when training ends
)

# Create a Trainer instance with checkpoints
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model using the Trainer instance
trainer.train()

# Save the final model
model.save_pretrained("./fine_tuned_model")
