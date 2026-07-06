import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class EmailResponder:
    def __init__(self, model_path="./trained_model"):
        # Fallback to the base model if fine-tuned model doesn't exist yet
        if not os.path.exists(model_path):
            print(f"Warning: '{model_path}' not found. Loading base model 'google/flan-t5-small' instead.")
            model_path = "google/flan-t5-small"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
    def generate_response(self, incoming_email):
        prompt = "Reply to email: " + incoming_email
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    responder = EmailResponder()
    test_email = "Can we schedule a meeting next week?"
    print(f"Incoming: {test_email}")
    print(f"Response: {responder.generate_response(test_email)}")
