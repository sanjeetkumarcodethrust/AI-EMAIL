import evaluate
import json
from inference import EmailResponder
import nltk
from sentence_transformers import SentenceTransformer, util

def get_rationale(semantic_score, rouge_score):
    if semantic_score > 0.85:
        if rouge_score > 0.5:
            return "Excellent. High semantic match and similar wording to the ground truth."
        else:
            return "Good. The meaning is highly accurate, even though the specific wording differs from the ground truth."
    elif semantic_score > 0.6:
        return "Acceptable. Captures the general gist but may miss some nuances of the ground truth."
    else:
        return "Poor. The generated response diverges significantly in meaning from the expected reply."

def evaluate_model():
    print("Loading test dataset...")
    test_data = []
    with open("test.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))
            
    print("Loading evaluation metrics (ROUGE & SentenceTransformers)...")
    rouge = evaluate.load("rouge")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing EmailResponder...")
    responder = EmailResponder()
    
    report = []
    total_semantic_score = 0.0
    
    print(f"Evaluating {len(test_data)} examples...")
    for idx, item in enumerate(test_data):
        if idx > 0 and idx % 10 == 0:
            print(f"Processed {idx} examples...")
            
        incoming = item["incoming_email"]
        true_response = item["response"]
        predicted_response = responder.generate_response(incoming)
        
        # Calculate individual ROUGE
        rouge_res = rouge.compute(predictions=[predicted_response], references=[[true_response]])
        rouge_l = rouge_res['rougeL']
        
        # Calculate Semantic Similarity
        emb1 = semantic_model.encode(predicted_response, convert_to_tensor=True)
        emb2 = semantic_model.encode(true_response, convert_to_tensor=True)
        semantic_score = util.cos_sim(emb1, emb2).item()
        
        total_semantic_score += semantic_score
        
        # Determine Rationale
        rationale = get_rationale(semantic_score, rouge_l)
        
        report.append({
            "incoming_email": incoming,
            "ground_truth": true_response,
            "generated_reply": predicted_response,
            "metrics": {
                "semantic_similarity": round(semantic_score, 4),
                "rouge_L": round(rouge_l, 4)
            },
            "rationale": rationale
        })
        
    overall_accuracy = total_semantic_score / len(test_data)
    
    print("\n--- Evaluation Results ---")
    print(f"Overall System Accuracy (Mean Semantic Similarity): {overall_accuracy * 100:.2f}%")
    print("A detailed per-response report has been saved to 'evaluation_report.json'.")
    
    with open("evaluation_report.json", "w") as f:
        json.dump({
            "overall_system_score": f"{overall_accuracy * 100:.2f}%",
            "per_response_evaluation": report
        }, f, indent=4)
        
    return overall_accuracy

if __name__ == "__main__":
    evaluate_model()
