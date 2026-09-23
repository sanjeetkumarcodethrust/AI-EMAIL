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

## Architecture & Generative AI Approach

To generate responses, this system uses **Fine-Tuning on a Generative LLM** (`google/flan-t5-small`, a seq2seq generative AI model). The system does not use classical ML classification, it generates the text output token-by-token.

### Justification of Trade-offs (Fine-Tuning vs. RAG / Few-Shot Prompting)

When designing this system, several Generative AI approaches were considered for "learning" from the dataset:

1. **RAG / Dynamic Few-Shot Prompting:**
   - *Pros:* Very flexible; if the dataset changes, the model instantly adapts without retraining.
   - *Cons:* Huge context window overhead for every single email processed. It requires sending the past emails as part of the prompt, which incurs high latency and potentially high API token costs if using an external LLM.

2. **Fine-Tuning a small LLM (The Chosen Approach):**
   - *Pros:* **Zero API Costs & Low Latency.** By fine-tuning `t5-small`, the model internalizes the tone, style, and structure of the dataset directly into its weights. It does not need past emails injected into its prompt at inference time, saving massive amounts of context overhead. It is fast enough to run locally on a CPU.
   - *Cons:* Requires upfront compute time to train. The knowledge is static; if the company completely changes its email style guidelines, the model must be retrained.

## Evaluation & Accuracy Measurement

Defining "accuracy" for generative AI is challenging because an **exact string match is far too strict**; a generated reply can be completely correct in meaning but use different wording than the ground truth.

To measure real quality, our evaluation system (`evaluate_model.py`) uses a composite metric approach:
1. **Semantic Similarity:** We use a dense embedding model (`sentence-transformers/all-MiniLM-L6-v2`) to convert both the generated reply and the ground truth reply into vector embeddings, and then calculate the cosine similarity between them. This proves that the *meaning* and *intent* of the generated reply matches the ground truth, regardless of the specific words chosen.
2. **ROUGE-L Score:** We measure the longest common subsequence to ensure that key phrases and structural elements (like names, dates, or specific entities) are preserved from the training distribution.

**Reporting:** The script provides:
- **Per-Response Scores:** Showing the exact semantic match score and ROUGE score for every individual test email, along with a generated rationale explaining *why* it received that score (e.g., "High semantic similarity indicates accurate meaning transfer despite phrasing differences").
- **Overall System Score:** An aggregated average accuracy across the entire test set.

## AI Tools Used in Development

In building this project, AI assistants (specifically Google's Gemini / DeepMind agents) were used to:
1. **Brainstorm Architecture:** Discussing the trade-offs between RAG, few-shot prompting, and fine-tuning an LLM to arrive at the lightweight `flan-t5-small` fine-tuning approach.
2. **Code Generation:** Writing boilerplate code for the Hugging Face `transformers` Trainer loop and the `gradio` web interface.
3. **Evaluation Strategy Design:** Helping to formulate the composite accuracy metric (Semantic Similarity + ROUGE-L) to solve the "exact string match is too strict" problem inherent in generative text evaluation.
4. **Synthetic Data Generation:** Writing the Python script with hand-authored templates to programmatically generate the clean dataset of 200 emails.

## Implementation Details

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
