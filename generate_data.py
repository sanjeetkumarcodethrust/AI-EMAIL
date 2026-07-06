import json
import random

def generate_synthetic_data(num_samples=200):
    intents = [
        ("meeting_request", ["Can we schedule a meeting next week?", "Are you available for a quick sync tomorrow?", "Let's catch up on the project status this Friday."], 
         ["Sure, let's schedule a meeting. Does Tuesday at 10 AM work for you?", "I am available tomorrow at 2 PM. Send an invite.", "Friday works for me. Looking forward to it."]),
        ("status_update", ["What is the current status of the AI project?", "Have we finished the latest sprint?", "Can you send me the Q3 report?"],
         ["The AI project is on track. I will send a detailed report shortly.", "The sprint is completed, and all features are deployed.", "I have attached the Q3 report to this email."]),
        ("support_ticket", ["My account is locked, can you help?", "I forgot my password, how do I reset it?", "The software is crashing on startup."],
         ["I have unlocked your account. Please try logging in again.", "You can reset your password using the 'Forgot Password' link on the login page.", "We are aware of the startup issue and are working on a patch."]),
        ("thank_you", ["Thanks for the help yesterday!", "Appreciate your hard work on this.", "Thank you for the quick response."],
         ["You're welcome! Let me know if you need anything else.", "Happy to help!", "Anytime! Have a great day."]),
        ("follow_up", ["Just following up on my previous email.", "Any updates on this?", "Did you get a chance to review the document?"],
         ["Thanks for following up. I will review it today and get back to you.", "No updates yet, I will let you know as soon as I have more information.", "I have reviewed it and left some comments in the document."])
    ]
    
    dataset = []
    for _ in range(num_samples):
        intent, user_prompts, responses = random.choice(intents)
        incoming_email = random.choice(user_prompts)
        
        # Add some slight variations to incoming email
        variation = random.choice(["", " Hi, ", " Hello there, ", " Dear team, "])
        sign_off = random.choice(["", " Thanks.", " Best,", " Cheers!"])
        incoming_email = f"{variation.strip()} {incoming_email} {sign_off.strip()}".strip()
        
        response = random.choice(responses)
        dataset.append({
            "incoming_email": incoming_email,
            "response": response
        })
        
    return dataset

if __name__ == "__main__":
    data = generate_synthetic_data(200)
    
    # Split into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    with open("train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples.")
