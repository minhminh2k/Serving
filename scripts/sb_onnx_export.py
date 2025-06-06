import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

huggingface_cached = "/home/duong.quang.minh/project/packtech-innovate/sunec/triton/huggingface_cached"

prompt = "Draw a dog"
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
)

tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir=huggingface_cached
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir=huggingface_cached
)

# Only get the decoder part of the VAE
vae.forward = vae.decode

torch.onnx.export(
    vae,
    (torch.randn(1, 4, 64, 64), False),
    "models/vae.onnx",
    input_names=["latent_sample", "return_dict"],
    output_names=["sample"],
    dynamic_axes={
        "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    opset_version=14,
)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

print("Text encoder input shape: ", text_input.input_ids.shape) # torch.Size([1, 77])

torch.onnx.export(
    text_encoder,
    (text_input.input_ids.to(torch.int32)),
    "models/encoder.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
    do_constant_folding=True,
)