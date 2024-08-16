# Enhance Agent
## Introduction
Enhance Agent is an LLM based agent to run various tools on images. It can zoom in on the image, upscale the image,
and describe the image for the user. It also takes advantage of whisper to provide a more interactive experience with voice commands.
For the image upscaling, it uses the EDSR model, for the image zooming, it uses the Florence model to generate the bounding boxes and then zooms in on the image. 
The description of the image is also generated by the Florence model. Then the Functionary v2.4 model is used to generate the text for the user and call the tools.

## Models Used
1. [Functionary](https://huggingface.co/meetkai/functionary-small-v2.4)
2. [Whisper](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
3. [Florence](https://huggingface.co/microsoft/Florence-2-base-ft)
4. [EDSR](https://huggingface.co/eugenesiow/edsr-base)

## Installation
1. Clone the repository.
```bash
git clone https://github.com/dwight-foster/EnhanceAgent.git
```
2. Download the functionary gguf and whisper gguf from the links above.
3. Install llama-cpp-python server via pip.
```bash
pip install llama-cpp-python[server]
```
4. Install whisper cpp using the instructions [here](https://github.com/ggerganov/whisper.cpp)
5. Install the other requirements using the requirements.txt file.
```bash
pip install -r requirements.txt
```

## Usage
This project was run on a macbook pro with 32GB of memory. It is running florence on cpu because of flash attention limitations. 
If you want to run florence on your gpu go into tools.py and change the code to run on the gpu. All the models minus Florence take up about 10GB of memory.
1. Run the llama-cpp server.
```bash
python3 -m llama_cpp.server --model weights/functionary-small-v2.4.Q4_0.gguf --chat_format functionary-v2 --hf_pretrained_model_name_or_path meetkai/functionary-small-v2.4 
```
2. Run the whisper server.
```bash
cd whisper.cpp
./server -m ../weights/ggml-medium-q5_0.bin
```
3. Run the run_gradio.py file.
```bash
python run_gradio.py
```
4. Open the link in your browser and upload an image.

## Demo

[![Demo](src/bourne.jpg)](https://youtu.be/yDaR5sffpvI)

## Limitations
1. The image upscaling is not very good. I used a smaller model to reduce the amount of memory used. 
2. The whisper model sometimes does not recognize the voice commands correctly.
3. All together the models can take a lot of memory and can be slow on a CPU.
4. The Florence model can sometimes generate incorrect bounding boxes.
5. The Functionary model can sometimes generate incorrect text and call the wrong tools.

## Future Work
1. Test different models for image upscaling.
2. Utilize the newer functionary models.
3. Switch to native llama cpp or ollama for the functionary models
4. Implement TTS 

## Acknowledgements
1. [Functionary](https://github.com/MeetKai/functionary)
2. [Whisper](https://github.com/ggerganov/whisper.cpp)
3. [LlamaCPP](https://github.com/abetlen/llama-cpp-python)
