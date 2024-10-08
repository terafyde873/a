import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, Trainer, TrainingArguments

# Set environment variables for multi-core processing
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust according to the number of CPU cores
os.environ["MKL_NUM_THREADS"] = "4"  # Adjust according to the number of CPU cores

# Set PyTorch to use all available cores
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))

# Define the path to the JSONL file
data_file = "mbpp.jsonl"

# Load tokenizer
model_name = "Salesforce/codegen-350M-mono"  # Updated to use the specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer if not already present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
if os.path.exists(data_file):
    dataset = load_dataset('json', data_files=data_file, split='train')
    print("Loaded dataset successfully.")
else:
    raise FileNotFoundError(f"File {data_file} does not exist.")

# Inspect the dataset to determine the column name
text_column_name = dataset.column_names[0]  # Assuming the text column is the first column

# Function to tokenize data
def tokenize_function(examples):
    return tokenizer(examples[text_column_name], padding="max_length", truncation=True)

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Setup Data Collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True for masked language modeling, False for causal LM
)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # Adjust as needed
    weight_decay=0.01,
    logging_first_step=True,
    load_best_model_at_end=True,
    warmup_steps=500,
    fp16=True,  # Ensure GPU support; set to False if not using GPU
    report_to="none",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Optional: Clean up intermediate files after training (uncomment if needed)
# if os.path.exists("./results"):
#     shutil.rmtree("./results")
# if os.path.exists("./logs"):
#     shutil.rmtree("./logs")
