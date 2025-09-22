from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
from gradio_ui import demo

app = FastAPI()

# @app.get("/")
# async def root():
#     return JSONResponse(content={"message": "Gradio app is running at /gradio"})

# # Gradio'ni FastAPI ga mount qilish
app = gr.mount_gradio_app(app, demo, path="/")
