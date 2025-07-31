from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import numpy as np
from diffusers import StableDiffusionPipeline 
from PIL import Image
import time

HUGGINGFACE_CACHED = "/mnt/nvme1n1/duong.quang.minh/huggingface_cached"
BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16, 
    variant="fp16",
    cache_dir=HUGGINGFACE_CACHED  
  ).to("cuda")


image = Image.open("/home/duong.quang.minh/project/Serving/assets/dog.png").convert("RGB").resize((1024, 1024)) # (1024, 1024, 3)
mask_image = Image.open("/home/duong.quang.minh/project/Serving/assets/mask.png").convert("RGB").resize((1024, 1024)) # (1024, 1024, 3)

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

start_time = time.time()

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

image.save("output.png")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds") # approx 3 seconds for 20 steps