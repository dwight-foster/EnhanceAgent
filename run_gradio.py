import openai
from utils import call_tool
from tools import Tools
import gradio as gr

client = openai.OpenAI(
    api_key="llama-cpp",  # can be anything
    base_url="http://localhost:8000/v1",  # NOTE: Replace with IP address and port of your llama-cpp-python server
)

tools=[{
            "type": "function",
            "function": {
                "name": "zoom",
                "description": "Zoom in on any objects image. You already have the image you don't need it. Could be faces, people, boats, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The description of the objects to detect. Must end in  a .",
                        }
                    },
                    "required": ["text"]
                }
            }

        },
        {
            "type": "function",
            "function": {
                "name": "describe",
                "description": "Describe the objects in an image. You already have the image you don't need it.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "upscale_image",
                "description": "Upscale an image by 4x. You already have the image you don't need it.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "zoom_out",
                "description": "Zoom out on an image. You already have the image you don't need it.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]

from gradio import ChatMessage

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            with gr.Accordion("Use Microphone", open=False):
                audio = gr.Microphone(type="filepath", waveform_options=gr.WaveformOptions(sample_rate=16000))
            clear = gr.Button("Clear")
        with gr.Column():
            image = gr.Image(type="filepath", height=550, width=550)

    tool = Tools()


    def chat_function(message, history):
        history.append(ChatMessage(role="user", content=message))
        history.append(ChatMessage(role="assistant", content=""))
        return history


    def user(user_message, history):

        return "", history + [[user_message, None]]


    def bot(message, history, image):

        prompt = []
        history = history or []
        for i in range(min(len(history), 10)):
            msg = history[i]
            prompt.append({"role": "user", "content": msg[0]})
            prompt.append({"role": "assistant", "content": msg[1]})
        prompt.append({"role": "system",
                       "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary."})
        if image is not None:
            prompt.append({"role": "system", "content": "The user uploaded an image."})
        else:
            prompt.append({"role": "system", "content": "The user did not upload an image."})
        prompt.append({"role": "user", "content": message})
        history.append([message, ""])
        bot_message = \
        client.chat.completions.create(model="llama", messages=prompt, tools=tools, tool_choice="auto").choices[
            0].message
        tool_calls = bot_message.tool_calls
        if tool_calls:
            image, bot_message = call_tool(client, tool_calls, bot_message, prompt, tool)

        history[-1][1] = bot_message.content
        return "", history, image


    image.upload(tool.upload_image, image, None, queue=False)
    image.change(tool.load_image, image, None, queue=False)
    image.clear(tool.clear, None, queue=False)
    msg.submit(bot, [msg, chatbot, image], [msg, chatbot, image], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    audio.stop_recording(tool.transcribe, [audio], [msg], queue=False).then(lambda: None, None, audio, queue=False)

demo.launch()

