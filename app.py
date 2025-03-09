import gradio as gr
from rag import get_response
import os


def answer_question(question: str):
    if not question:
        return "Please enter question"
    try:
        response = get_response(question)
        
        return response
    except Exception as e:
        return "Something Went Wrong"


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Enter your question here"),
    outputs=gr.Textbox(lines=8, label="Answer"),
    title="RAG Chatbot",
    description="Ask any question and get answer from RAG chatbot",
)

if __name__ == "__main__":
    demo.launch()
