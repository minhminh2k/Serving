import os
import onnx
import shutil
import torch
from inspect import signature
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import StableDiffusionPipeline, StableDiffusionXLInpaintPipeline
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionInpaintPipeline
from pathlib import Path

from src.models.unet_2d_condition import UNet2DConditionModel_SDXL
from packaging import version
from torch.onnx import export


from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.utils import USE_PEFT_BACKEND

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
    export(
        model,
        model_args,
        f=output_path,
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

HUGGINGFACE_CACHED = "/mnt/nvme1n1/duong.quang.minh/huggingface_cached"
BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet2DConditionModel_SDXL.from_pretrained(
    BASE_MODEL,
    subfolder="unet",
    cache_dir=HUGGINGFACE_CACHED,
    torch_dtype=torch.float16,
).to(device)

# pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float16,
#     cache_dir=HUGGINGFACE_CACHED,
# ).to(device)

# print(isinstance(pipeline, StableDiffusionXLInpaintPipeline))
# print(isinstance(pipeline, FromSingleFileMixin))
# print(isinstance(pipeline, IPAdapterMixin))
# print(isinstance(pipeline, StableDiffusionXLLoraLoaderMixin))
# print(isinstance(pipeline, TextualInversionLoaderMixin))

# print(USE_PEFT_BACKEND)


dtype = torch.float16

batch_size = 1
height = width = 128
num_tokens = 77
cross_attention_dim = model.config.cross_attention_dim 

sample = torch.randn(1, 9, 128, 128, dtype=dtype).to(device)
timestep = torch.tensor([1.0], dtype=dtype).to(device)
encoder_hidden_states = torch.randn(1, 77, 2048, dtype=dtype).to(device)
added_cond_kwargs = {
    "text_embeds": torch.randn(1, 1280, dtype=dtype).to(device),
    "time_ids": torch.randint(0, 1000, (1, 6)).to(dtype).to(device),
}

with torch.no_grad():
    output = model(
        sample=sample,
        timestep=timestep,
        timestep_cond=None,
        encoder_hidden_states=encoder_hidden_states,
        # added_cond_kwargs=added_cond_kwargs
        text_embeds=torch.randn(1, 1280, dtype=dtype).to(device),
        time_ids=torch.randint(0, 1000, (1, 6)).to(dtype).to(device),
    )
    print(output)  # 1, 4, 128, 128

    onnx_export(
        model,
        model_args=(
            torch.randn(1, 9, 128, 128).to(device=device, dtype=dtype),
            torch.tensor([1.0], dtype=dtype).to(device),
            torch.randn(1, 77, 2048).to(device=device, dtype=dtype),
            torch.randn(1, 1280, dtype=dtype).to(device),
            torch.randint(0, 100, (1, 6)).to(dtype).to(device),
            False,
        ),
        output_path="sdxl/unet_sdxl.onnx",
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "return_dict"],

        output_names=["out_sample"], 
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "text_embeds": {0: "batch"},
            "time_ids": {0: "batch"},
            "out_sample": {0: "batch"},
        },
        opset=14,
        use_external_data_format=True,
    )

    unet_model_path = "sdxl/unet_sdxl.onnx"
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
        location="model.onnx_data",
        convert_attribute=False,
    )