from typing import Dict

from pathlib import Path

import time
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from optimum.exporters.onnx import export_models, validate_models_outputs
from transformers.models.clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, ViTOnnxConfig
from diffusers import StableDiffusionPipeline, StableDiffusionXLInpaintPipeline

from optimum.onnxruntime import ORTStableDiffusionXLInpaintPipeline
from optimum.onnxruntime import ORTStableDiffusionPipeline
from optimum.onnxruntime import ORTModelForCustomTasks

from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline

from src import CLIPTextModelWithProjectionOnnxConfig, CLIPVisionWithProjectionOnnxConfig, CLIPVisionOnnxConfig

HUGGINGFACE_CACHED = "/home/duong.quang.minh/project/packtech-innovate/sunec/triton/assets"
BASE_MODEL = 'yisol/IDM-VTON'
OUTPUT_DIR = "models"

# idm_pipeline_model = ORTStableDiffusionXLInpaintPipeline.from_pretrained(
#     base_path,
#     # revision="onnx",
#     provider="CUDAExecutionProvider",
#     torch_dtype="float16",
#     safety_checker=None,
#     cache_dir=HUGGINGFACE_CACHED,
#     export=True,
# )

text_model = CLIPTextModelWithProjection.from_pretrained(
    BASE_MODEL,
    subfolder='text_encoder_2',
    torch_dtype=torch.float32,
    cache_dir=HUGGINGFACE_CACHED,
)
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    BASE_MODEL,
    subfolder='image_encoder',
    torch_dtype=torch.float32,
    cache_dir=HUGGINGFACE_CACHED,
)

def clip_vision_export(
    vision_model: CLIPVisionModelWithProjection,
) -> tuple:
    """
    Export the CLIP vision model to ONNX format.
    """
    return export_models(
        models_and_onnx_configs={"vision_model": (vision_model, CLIPVisionWithProjectionOnnxConfig(vision_model.config))},
        output_dir=OUTPUT_DIR,
    ) # [['pixel_values']], [['image_embeds', 'last_hidden_state']]

if __name__ == "__main__":

    start_time = time.time()
    elements, onnx_outputs = export_models(
        models_and_onnx_configs={
            "text_model": (text_model, CLIPTextModelWithProjectionOnnxConfig(text_model.config)),
            "vision_model": (vision_model, CLIPVisionWithProjectionOnnxConfig(vision_model.config)),
        },
        output_dir="models",
    )
    end_time = time.time()
    # print("First element:", elements)  # [['input_ids'], ['pixel_values']]
    # print("ONNX outputs:", onnx_outputs)  # [['text_embeds'], ['image_embeds', 'last_hidden_state']]
    print(f"Export completed in {end_time - start_time:.2f} seconds.")

    start_time = time.time()

    validate_models_outputs(
        models_and_onnx_configs={"text_model": (text_model, CLIPTextModelWithProjectionOnnxConfig(text_model.config))},
        onnx_named_outputs=[['text_embeds']],
        output_dir=Path(OUTPUT_DIR),
        onnx_files_subpaths=["text_model.onnx"],
        device="cpu",
    )

    validate_models_outputs(
        models_and_onnx_configs={"vision_model": (vision_model, CLIPVisionWithProjectionOnnxConfig(vision_model.config))},
        onnx_named_outputs=[['image_embeds', 'last_hidden_state']], # onnx_outputs,
        output_dir=Path(OUTPUT_DIR),
        onnx_files_subpaths=["vision_model.onnx"],
        device="cpu",
    )

    end_time = time.time()

    print(f"Validation completed in {end_time - start_time:.2f} seconds.")


    # 

    # vision_model = CLIPVisionModelWithProjection.from_pretrained(
    #     BASE_MODEL,
    #     subfolder='image_encoder',
    #     torch_dtype=torch.float32,
    #     cache_dir=HUGGINGFACE_CACHED,
    # )

    # import os
    # import json
    # config_path = os.path.join(OUTPUT_DIR, "config.json")
    # if not os.path.exists(config_path):
    #     with open(config_path, "w") as f:
    #         json.dump(vision_model.config.to_dict(), f)

    # start_time = time.time()

    # ort_model = ORTModelForCustomTasks.from_pretrained(
    #     OUTPUT_DIR, config=vision_model.config,
    #     libriary_name="transformers",
    # )

    # pixel_values = torch.zeros((10, 3, 224, 224))
    # ort_model(pixel_values=pixel_values)

    # end_time = time.time()
    # print(f"ORTModelForCustomTasks completed in {end_time - start_time:.2f} seconds.")