from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
from super_image import MdsrModel, ImageLoader
from unittest.mock import patch
from PIL import Image
import json
import requests
import torch
import os


class Tools():
    def __init__(self):
        # Workaround only for Macs. Cuda does not need this.
        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
            if not str(filename).endswith("/modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        self.upscale = MdsrModel.from_pretrained('eugenesiow/mdsr-bam', scale=2).to("mps")

        self.image = None
        self.path = None
        self.url = "http://127.0.0.1:8080/inference"
        self.original_image = None

    def upload_image(self, image_path):
        self.load_image(image_path)
        self.original_image = self.image

    def load_image(self, image_path):
        self.path = image_path
        self.image = Image.open(image_path)

    def clear(self):
        self.path = None
        self.image = None

    def run_example(self, task_prompt, text_input=None):

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt,
                                                               image_size=(self.image.width, self.image.height))

        return parsed_answer

    def zoom(self, text: str):
        """Zoom in on an image."""
        if self.image is None:
            return self.image, json.dumps({"result": "Please upload an image first."})

        results = self.run_example("<CAPTION_TO_PHRASE_GROUNDING>", text)
        box = results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'][0]
        x, y, width, height = box[0], box[1], box[2], box[3]
        cropped_image = self.image.crop([x, y, width, height])
        return cropped_image, json.dumps({"result": f"Successfully zoomed in on {text}."})

    def zoom_out(self):
        """Zoom out on an image."""
        if self.image is None:
            return self.image, json.dumps({"result": "Please upload an image first."})

        self.image = self.original_image
        return self.image, json.dumps({"result": "Successfully zoomed out."})

    def describe(self):
        """Describe an image."""
        if self.image is None:
            return self.image, json.dumps({"result": "Please upload an image first."})

        results = self.run_example("<DETAILED_CAPTION>", "")
        return self.image, json.dumps({"result": results})

    def upscale_image(self):
        """Upscale an image by 2x."""
        if self.image is None:
            return self.image, json.dumps({"result": "Please upload an image first."})
        inputs = ImageLoader().load_image(self.image)
        with torch.no_grad():
            pred = self.upscale(inputs.to("mps"))
        new_path = self.path.split(".")[0] + "_upscaled.jpeg"
        ImageLoader().save_image(pred.cpu(), new_path)
        self.load_image(new_path)

        return self.image, json.dumps({"result": "Image successfully upscaled."})

    def transcribe(self, audio_path):
        """"Transcribe an audio file."""
        headers = {
            'Content-Type': 'audio/wav',
        }
        files = {
            'file': audio_path,
            'temperature': '0.0',
            'temperature_inc': '0.2',
            'response_format': 'json'
        }

        response = requests.post(self.url, files=files, data=headers)
        return response.json()['text']
