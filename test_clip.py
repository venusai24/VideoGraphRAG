import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image = Image.new('RGB', (224, 224), color = 'red')
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model.get_image_features(**inputs)

print("Type of outputs:", type(outputs))
if isinstance(outputs, torch.Tensor):
    print("outputs is a Tensor")
elif hasattr(outputs, 'image_embeds'):
    print("outputs has image_embeds")
elif hasattr(outputs, 'pooler_output'):
    print("outputs has pooler_output")
print("Attributes:", dir(outputs))

