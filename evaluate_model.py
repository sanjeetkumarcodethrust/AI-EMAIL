import evaluate
import json
from inference import EmailResponder
import nltk

def evaluate_model():
    print("Loading test dataset...")
    test_data = []
    with open("test.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))
            
    print("Loading evaluation metrics (ROUGE & BLEU)...")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    
    print("Initializing model...")
    responder = EmailResponder()
    
    predictions = []
    references = []
    
    print(f"Evaluating {len(test_data)} examples...")
    for idx, item in enumerate(test_data):
        if idx > 0 and idx % 10 == 0:
            print(f"Processed {idx} examples...")
            
        incoming = item["incoming_email"]
        true_response = item["response"]
        
        predicted_response = responder.generate_response(incoming)
        
        predictions.append(predicted_response)
        references.append([true_response]) # sacrebleu expects list of references
        
    print("Calculating scores...")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)
    
    print("\n--- Evaluation Results ---")
    print("ROUGE Scores (Measures overlap in content):")
    for key, value in rouge_results.items():
        print(f"  {key}: {value:.4f}")
        
    print(f"\nBLEU Score (Measures exact word matches): {bleu_results['score']:.2f}")
    
    return rouge_results, bleu_results

if __name__ == "__main__":
    evaluate_model()
