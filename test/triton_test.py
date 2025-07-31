import os
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *
import torch

def main():
    client = httpclient.InferenceServerClient(url="0.0.0.0:8111")

    # sample  = httpclient.InferInput(
    #     "sample", [1, 3, 1024, 1024], "FP32"
    # )
    # sample.set_data_from_numpy(
    #     np.random.rand(1, 3, 1024, 1024).astype(np.float32)  # Simulated input
    # )
    
    # output_img = httpclient.InferRequestedOutput("latent_sample")

    # query_response = client.infer(
    #     model_name="vae_encoder", inputs=[sample], outputs=[output_img]
    # )

    # image = query_response.as_numpy("latent_sample")
    # print(f"Image shape: {image.shape}, Image dtype: {image.dtype}, max value: {image.max()}, min value: {image.min()}")

    sample = httpclient.InferInput(
        "sample", [1, 9, 128, 128], "FP16"
    )
    sample.set_data_from_numpy(
        np.random.rand(1, 9, 128, 128).astype(np.float16)  # Simulated input
    )
    
    timestep = httpclient.InferInput(
        "timestep", [1], "FP16"
    )
    timestep.set_data_from_numpy(np.array([101], dtype=np.float16))
    
    encoder_hidden_states = httpclient.InferInput("encoder_hidden_states", [1, 77, 2048], "FP16")
    encoder_hidden_states.set_data_from_numpy(
        np.random.rand(1, 77, 2048).astype(np.float16)  # Simulated input
    )
    
    text_embeds = httpclient.InferInput("text_embeds", [1, 1280], "FP16")
    text_embeds.set_data_from_numpy(
        np.random.rand(1, 1280).astype(np.float16))  # Simulated input
    
    time_ids = httpclient.InferInput("time_ids", [1, 6], "FP16")
    time_ids.set_data_from_numpy(np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float16))
    
    output = httpclient.InferRequestedOutput("out_sample")
    
    response = client.infer(
        model_name="unet",
        inputs=[sample, timestep, encoder_hidden_states, text_embeds, time_ids],
        outputs=[output]
    )
    
    out_sample = response.as_numpy("out_sample")
    print(f"Output shape: {out_sample.shape}, Output dtype: {out_sample.dtype}, max value: {out_sample.max()}, min value: {out_sample.min()}")
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Processing time", end - start)