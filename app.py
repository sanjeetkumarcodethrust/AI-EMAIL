import gradio as gr
from inference import EmailResponder

# Initialize the responder globally
try:
    responder = EmailResponder()
except Exception as e:
    print(f"Error initializing model: {e}")
    responder = None

def suggest_reply(incoming_email):
    if responder is None:
        return "Error: Model could not be loaded."
    if not incoming_email.strip():
        return "Please enter an email."
    
    return responder.generate_response(incoming_email)

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Email Suggested-Response System")
    gr.Markdown("This system uses a sequence-to-sequence model to suggest a reply for an incoming email.")
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(
                lines=5, 
                label="Incoming Email",
                placeholder="e.g. Can we schedule a meeting next week?"
            )
            submit_btn = gr.Button("Generate Reply", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                lines=5,
                label="Suggested AI Reply",
                interactive=False
            )
            
    submit_btn.click(
        fn=suggest_reply,
        inputs=email_input,
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            "Can we schedule a meeting next week?",
            "What is the current status of the AI project?",
            "My account is locked, can you help?",
            "Thanks for the help yesterday!"
        ],
        inputs=email_input
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
