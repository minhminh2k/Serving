import time
import torch
import numpy as np
import tritonclient.http
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionPipeline

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


model_name = "sdxl_inpaint"
url = "0.0.0.0:8111"
model_version = "1"
batch_size = 1

# model input params

img_url = Image.open("/home/duong.quang.minh/project/Serving/assets/dog.png")
mask_url = Image.open("/home/duong.quang.minh/project/Serving/assets/mask.png")

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
negative_prompt = "NONE"
samples = 1 # no.of images to generate
scheduler = "EulerDiscreteScheduler"
steps = 20
guidance_scale = 8.0
seed = 100042

start_time = time.time()

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

# Input placeholder
image_in = tritonclient.http.InferInput(name="IMAGE", shape=(batch_size, 1024, 1024, 3), datatype="FP16")
mask_image_in = tritonclient.http.InferInput(name="MASK_IMAGE", shape=(batch_size, 1024, 1024, 3), datatype="FP16")
prompt_in = tritonclient.http.InferInput(name="PROMPT", shape=(batch_size,), datatype="BYTES")
negative_prompt_in = tritonclient.http.InferInput(name="NEGATIVE_PROMPT", shape=(batch_size,), datatype="BYTES")
samples_in = tritonclient.http.InferInput("SAMPLES", (batch_size, ), "INT32")
scheduler_in = tritonclient.http.InferInput(name="SCHEDULER", shape=(batch_size,), datatype="BYTES")
steps_in = tritonclient.http.InferInput("STEPS", (batch_size, ), "INT32")
guidance_scale_in = tritonclient.http.InferInput("GUIDANCE_SCALE", (batch_size, ), "FP16")
seed_in = tritonclient.http.InferInput("SEED", (batch_size, ), "INT64")

images = tritonclient.http.InferRequestedOutput(name="IMAGES", binary_data=False)

# Setting inputs
image_in.set_data_from_numpy(np.expand_dims(np.array(image, dtype=np.float16), axis=0))
mask_image_in.set_data_from_numpy(np.expand_dims(np.array(mask_image, dtype=np.float16), axis=0))
prompt_in.set_data_from_numpy(np.asarray([prompt] * batch_size, dtype=object))
negative_prompt_in.set_data_from_numpy(np.asarray([negative_prompt] * batch_size, dtype=object))
samples_in.set_data_from_numpy(np.asarray([samples], dtype=np.int32))
scheduler_in.set_data_from_numpy(np.asarray([scheduler] * batch_size, dtype=object))
steps_in.set_data_from_numpy(np.asarray([steps], dtype=np.int32))
guidance_scale_in.set_data_from_numpy(np.asarray([guidance_scale], dtype=np.float16))
seed_in.set_data_from_numpy(np.asarray([seed], dtype=np.int64))

response = triton_client.infer(
    model_name=model_name, model_version=model_version, 
    inputs=[image_in, mask_image_in, prompt_in,negative_prompt_in,samples_in,scheduler_in,steps_in,guidance_scale_in,seed_in], 
    outputs=[images]
)

images = response.as_numpy("IMAGES")

if images.ndim == 3:
    images = images[None, ...]
images = (images * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]


rows = 1 # change according to no.of samples 
cols = 1 # change according to no.of samples
# rows * cols == no.of samples

output_image = image_grid(pil_images, rows, cols)

output_image.save("output/output.png")

end_time = time.time()

print("Processing time:", end_time - start_time)
