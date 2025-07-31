import argparse
import os
import time
import shutil
from pathlib import Path

import onnx
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel


@torch.no_grad()
def inference_models(
    model_path: str, 
    huggingface_cached: str,
    fp16: bool = False
):
    dtype = torch.float16 if fp16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} with dtype {dtype} on device {device}")
        
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        cache_dir=huggingface_cached
    ).to(device)
    
    start_time = time.time()
    
    prompt = "A fantasy landscape, trending on artstation"
    
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds") # fp16: 1.31s + fp32: 3.3s
    
    image.save("inference_output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default= "CompVis/stable-diffusion-v1-4",
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--huggingface_cached",
        type=str,
        default="/mnt/nvme1n1/duong.quang.minh/packtech_assets/triton/huggingface_cached",
        help="Path to the huggingface cached path.",
    )
    
    parser.add_argument(
        "--opset",
        default=16,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    inference_models(args.model_path, args.huggingface_cached, args.fp16)
    