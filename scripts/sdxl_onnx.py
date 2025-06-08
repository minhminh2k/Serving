import argparse
import os
import shutil
from pathlib import Path

import time
import onnx
import torch
from packaging import version
from torch.onnx import export

from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from diffusers import StableDiffusionXLInpaintPipeline

from src.models.modeling_clip import CLIPTextModel as CLIPTextModel_SDXL
from src.models.modeling_clip import CLIPTextModelWithProjection as CLIPTextModelWithProjection_SDXL

from src.models.wrapped import WrappedTextEncoder
from diffusers import AutoencoderKL
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from optimum.exporters.onnx import export_models, validate_models_outputs
from src import CLIPTextModelWithProjectionOnnxConfig, CLIPVisionWithProjectionOnnxConfig
from src.models.unet_2d_condition import UNet2DConditionModel_SDXL

HUGGINGFACE_CACHED = "/mnt/nvme1n1/duong.quang.minh/huggingface_cached"
BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
OUTPUT_DIR = "sdxl/"


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )

@torch.no_grad()
def convert_models(
    output_path: str, 
    opset: int, 
    fp16: bool = False
):
    print("Convert to fp16: ", fp16)
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"

    pipeline: StableDiffusionXLInpaintPipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype, 
        # variant="fp16",
        cache_dir=HUGGINGFACE_CACHED,
        use_safetensors=True
    ).to(device)

    output_path = Path(output_path)
     
    # TEXT ENCODER 1
    pipeline.text_encoder = CLIPTextModel_SDXL.from_pretrained(
        BASE_MODEL,
        subfolder='text_encoder',
        torch_dtype=dtype,
        cache_dir=HUGGINGFACE_CACHED,
        use_safetensors=True,
    ).to(device)
    
    # text_encoder = WrappedTextEncoder(pipeline.text_encoder).to(device)
    
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    onnx_export(
        pipeline.text_encoder,
        model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output", "hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    del pipeline.text_encoder

    # TEXT ENCODER 2
    num_tokens = pipeline.text_encoder_2.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder_2.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # text_encoder_2 = pipeline.text_encoder_2
    text_encoder_2 = CLIPTextModelWithProjection_SDXL.from_pretrained(
        BASE_MODEL,
        subfolder='text_encoder_2',
        torch_dtype=torch.float32,
        cache_dir=HUGGINGFACE_CACHED,
    )
    
    image_encoder = pipeline.image_encoder

    if image_encoder is not None:
        start_time = time.time()
        elements, onnx_outputs = export_models(
            models_and_onnx_configs={
                "text_encoder_2": (text_encoder_2, CLIPTextModelWithProjectionOnnxConfig(text_encoder_2.config)),
                "image_encoder": (image_encoder, CLIPVisionWithProjectionOnnxConfig(image_encoder.config)),
            },
            output_dir=OUTPUT_DIR,
            # dtype="fp16" if fp16 else "fp32",
            device=device
        )
        end_time = time.time()
        # Elements: [['input_ids'], ['pixel_values']]
        # ONNX outputs: [['text_embeds'], ['image_embeds', 'last_hidden_state']]
        print(f"Export completed in {end_time - start_time:.2f} seconds.")

        start_time = time.time()

        validate_models_outputs(
            models_and_onnx_configs={"text_encoder_2": (text_encoder_2, CLIPTextModelWithProjectionOnnxConfig(text_encoder_2.config))},
            onnx_named_outputs=[['text_embeds']],
            output_dir=Path(OUTPUT_DIR),
            onnx_files_subpaths=["text_encoder_2.onnx"],
            device=device,
        )

        validate_models_outputs(
            models_and_onnx_configs={"image_encoder": (image_encoder, CLIPVisionWithProjectionOnnxConfig(image_encoder.config))},
            onnx_named_outputs=[['image_embeds', 'last_hidden_state']],
            output_dir=Path(OUTPUT_DIR),
            onnx_files_subpaths=["image_encoder.onnx"],
            device=device,
        )
    else:
        start_time = time.time()
        elements, onnx_outputs = export_models(
            models_and_onnx_configs={
                "text_encoder_2": (text_encoder_2, CLIPTextModelWithProjectionOnnxConfig(text_encoder_2.config)),
            },
            output_dir=OUTPUT_DIR,
            dtype="fp16",
            device=device
        )
        end_time = time.time()
        print(f"Export completed in {end_time - start_time:.2f} seconds.")

        validate_models_outputs(
            models_and_onnx_configs={"text_encoder_2": (text_encoder_2, CLIPTextModelWithProjectionOnnxConfig(text_encoder_2.config))},
            onnx_named_outputs=[['text_embeds']],
            output_dir=Path(OUTPUT_DIR),
            onnx_files_subpaths=["text_encoder_2.onnx"],
            # device=device,
        )

    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds.")

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    cross_attention_dim = pipeline.unet.config.cross_attention_dim
    text_hidden_size = pipeline.text_encoder_2.config.hidden_size

    unet_path = output_path / "unet" / "model.onnx"

    pipeline.unet = UNet2DConditionModel_SDXL.from_pretrained(
        BASE_MODEL,
        subfolder="unet",
        cache_dir=HUGGINGFACE_CACHED,
        torch_dtype=dtype,
    ).to(device)

    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, 77, cross_attention_dim).to(device=device, dtype=dtype),
            torch.randn(2, text_hidden_size, dtype=dtype).to(device),
            torch.randint(0, 100, (2, 6)).to(dtype).to(device),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "text_embeds": {0: "batch"},
            "time_ids": {0: "batch"},
            "out_sample": {0: "batch"},
        },
        opset=opset,
        use_external_data_format=True, # UNet is > 2GB, so the weights need to be split
    )

    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae
    

    # feature_extractor = pipeline.feature_extractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(OUTPUT_DIR, args.opset, args.fp16)