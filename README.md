# AI Email Suggested-Response System

This is a complete, local machine learning system that takes an incoming email and generates a contextually appropriate suggested reply. It is built using Python, the Hugging Face `transformers` library, and `gradio` for the web interface.

## Dataset Ownership & Representativeness

**Source:** The dataset for this project was **synthetically generated** using the included `generate_data.py` script. The script uses a hand-authored, template-based approach to create realistic pairs of incoming emails and their corresponding replies.

**Why it is representative:** 
Real-world corporate email datasets (like the Enron dataset) are often heavily unstructured, contain sensitive PII, and require massive amounts of cleaning before they can be mapped directly to a simple "incoming -> response" format. 

By synthetically generating the data, we ensure that the dataset is highly focused and clean. The data is representative because it models the top 5 most common workplace email intents:
1. **Meeting Requests** ("Are you available for a quick sync tomorrow?")
2. **Status Updates** ("What is the current status of the AI project?")
3. **Support Tickets / Troubleshooting** ("My account is locked, can you help?")
4. **Follow-ups** ("Just following up on my previous email.")
5. **Polite Acknowledgments** ("Thanks for the help yesterday!")

The generation script also injects natural variations in greetings (e.g., "Hi", "Hello there") and sign-offs (e.g., "Thanks", "Cheers!") to accurately mimic the variability found in real human communication.

## Architecture

1. **Dataset Generation (`generate_data.py`)**: Creates `train.jsonl` and `test.jsonl`.
2. **Model Fine-Tuning (`train.py`)**: Fine-tunes the `google/flan-t5-small` sequence-to-sequence model on the synthetic dataset to learn the mapping from an incoming email to a professional response.
3. **Inference (`inference.py`)**: Handles loading the trained weights and running generation.
4. **Evaluation (`evaluate_model.py`)**: Evaluates the model's performance on the unseen test set using standard NLP metrics (ROUGE and BLEU).
5. **Web Interface (`app.py`)**: Provides a beautiful, interactive Gradio interface to test the model.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate Data:**
   ```bash
   python generate_data.py
   ```
3. **Train the Model:**
   ```bash
   python train.py
   ```
4. **Evaluate the Model:**
   ```bash
   python evaluate_model.py
   ```
5. **Launch the UI:**
   ```bash
   python app.py
   ```
