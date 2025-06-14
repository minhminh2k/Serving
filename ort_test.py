from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch

from diffusers import StableDiffusionPipeline 

HUGGINGFACE_CACHED = "/mnt/nvme1n1/duong.quang.minh/huggingface_cached"
BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16, 
    # variant="fp16"
    cache_dir=HUGGINGFACE_CACHED  
  ).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

# print(type(image))
image.save("output.png")