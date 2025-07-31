import os
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *

OUTPUT_PATH = "assets/outputs"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

OUTPUT_SB_PATH = os.path.join(OUTPUT_PATH, "stable_diffusion_v1_4")

if not os.path.exists(OUTPUT_SB_PATH):
    os.makedirs(OUTPUT_SB_PATH)

def main():
    client = httpclient.InferenceServerClient(url="0.0.0.0:8111")

    prompt = "Pikachu with a red hat"
    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

    input_text = httpclient.InferInput(
        "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
    )
    input_text.set_data_from_numpy(text_obj)

    output_img = httpclient.InferRequestedOutput("generated_image")

    query_response = client.infer(
        model_name="pipeline", inputs=[input_text], outputs=[output_img]
    )

    image = query_response.as_numpy("generated_image")
    im = Image.fromarray(np.squeeze(image.astype(np.uint8)))
    im.save(os.path.join(OUTPUT_SB_PATH, "pikachu.png"))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Processing time", end - start)