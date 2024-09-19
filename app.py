from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Example dataset with a small amount of text data
data = {
    "text": [
        "I love machine learning.",
        "Fine-tuning models is very useful.",
        "Hugging Face makes it easy to use transformers.",
        "GPT models are powerful for generating text.",
        "This is a small dataset for demonstration."
    ]
}

# Create a Dataset
dataset = Dataset.from_dict(data)

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token to be the same as the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize and return the input_ids and attention_mask
    encodings = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=50)
    # Set the labels to be the same as the input_ids
    encodings['labels'] = encodings['input_ids']
    return encodings

# Map the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Adjust based on your CPU capabilities
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=1,
    save_steps=10,
)

# Create the Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start the fine-tuning process
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt")
tokenizer.save_pretrained("./fine-tuned-gpt")
