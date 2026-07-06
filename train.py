import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def main():
    model_id = "google/flan-t5-small"
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Load data
    print("Loading datasets...")
    dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})
    
    def preprocess_function(examples):
        inputs = ["Reply to email: " + email for email in examples["incoming_email"]]
        targets = examples["response"]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        
        # Tokenize targets
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        push_to_hub=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model to ./trained_model")
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")

if __name__ == "__main__":
    main()
