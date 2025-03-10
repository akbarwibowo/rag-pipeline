import gradio as gr
from rag import get_response
import os


def answer_question(question: str) -> str:
    """
    Answer a given question using a response generation system.
    This function processes a question and returns an appropriate response. If the question
    is empty, it returns a prompt asking for a question. If an error occurs during
    processing, it returns an error message.
    Args:
        question (str): The question to be answered.
    Returns:
        str: The response to the question, an error message if processing fails,
            or a prompt if no question is provided.
    """
    if not question:
        return "Please enter question"
    try:
        response = get_response(question)
        
        return response
    except Exception as e:
        return "Something Went Wrong"


# Create a Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Enter your question here"),
    outputs=gr.Textbox(lines=8, label="Answer"),
    title="RAG Chatbot",
    description="Ask any question and get answer from RAG chatbot",
)

if __name__ == "__main__":
    demo.launch()
