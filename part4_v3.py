import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from collections import defaultdict
from PIL import Image
import requests
import pandas as pd

# Set up model and processor
MODEL_CHECKPOINT = "Qwen/Qwen2-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Wrap the original model to insert profiling hooks
class ProfilingQwen2VL(Qwen2VLForConditionalGeneration):
    def forward(self, *args, **kwargs):
        # Visual embedding layer
        with record_function("Visual Embedding Layer"):
            if kwargs.get("pixel_values", None) is not None:
                pixel_values = kwargs["pixel_values"].to(self.device)
                
                # Ensure grid_thw is a 2D tensor with shape (batch_size, 3)
                grid_thw = kwargs.get("image_grid_thw", torch.tensor([[1, 92, 62]]).to(self.device))

                # Call the visual embedding layer
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
                kwargs["inputs_embeds"] = image_embeds
        
        # Text embedding layer
        with record_function("Text Embedding Layer"):
            if kwargs.get("input_ids", None) is not None:
                kwargs["inputs_embeds"] = self.model.embed_tokens(kwargs["input_ids"])

        # Encoder
        with record_function("Encoder"):
            encoder_outputs = self.model(
                inputs_embeds=kwargs["inputs_embeds"],
                attention_mask=kwargs.get("attention_mask"),
                return_dict=kwargs.get("return_dict", True)
            )
        
        # Decoder
        with record_function("Decoder"):
            outputs = self.lm_head(encoder_outputs.last_hidden_state)

        return outputs

# Load the wrapped model
model = ProfilingQwen2VL.from_pretrained(MODEL_CHECKPOINT).to(device)
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)

# Prepare inputs
def prepare_inputs():
    text = "Describe this image."
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"

    # Load an image for vision input
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process input with AutoProcessor
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    # Add grid dimensions for the image
    inputs["image_grid_thw"] = torch.tensor([[1, 92, 62]])  # Ensure correct shape: (1, 3)
    return inputs.to(device)

inputs = prepare_inputs()

# Profiling function
def run_model_with_profiling():
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"]
            )

    return prof

# Run the profiler
prof = run_model_with_profiling()

# Convert profiling data to a pandas DataFrame
prof_data = []
for event in prof.key_averages():
    prof_data.append({
        "Name": event.key,
        "CPU Time Total (us)": event.cpu_time_total,
    })

df = pd.DataFrame(prof_data)

# Save the DataFrame to an Excel file
output_file = "./profiling_results.xlsx"
df.to_excel(output_file, index=False)

output_file
