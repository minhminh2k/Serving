
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import os
import onnx
import torch
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 1

HUGGINGFACE_CACHED = "/mnt/nvme1n1/duong.quang.minh/packtech_assets/triton/huggingface_cached"
OUTPUT_FOLDER = "model_repository/stable_diffusion_v1_4"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    subfolder="vae", use_auth_token=True,
    cache_dir=HUGGINGFACE_CACHED
).to(device)

tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir=HUGGINGFACE_CACHED
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir=HUGGINGFACE_CACHED
).to(device)

# Only get the decoder part of the VAE
vae.forward = vae.decode

def checking_onnx_model(model_path: str, providers: list = ["CUDAExecutionProvider"]):
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model {model_path} is valid.")
        
        sess = ort.InferenceSession(model_path, providers=providers)
        for node in sess.get_modelmeta().custom_metadata_map.items():
            print(node)
        
    except Exception as e:
        print(f"Error checking ONNX model {model_path}: {e}")

def inference_session(model_path: str, providers: list = ["CUDAExecutionProvider"]):
    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        print(f"ONNX model {model_path} loaded successfully.")
        return sess
    except Exception as e:
        print(f"Error loading ONNX model {model_path}: {e}")
        return None

def inference_onnx_model(model_path: str, inputs: dict, providers: list = ["CUDAExecutionProvider"]):
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
        
        for inp in sess.get_inputs():
            print(f"Input name: {inp.name} --- Shape: {inp.shape} --- Type: {inp.type}")

        prepared_inputs = {
            k: (v.numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
            if isinstance(v, (np.ndarray, list, float, int)) or hasattr(v, "numpy")
        }

        outputs = sess.run(None, prepared_inputs)
        print(f"Output names: {[o.name for o in sess.get_outputs()]}")
        print(f"Outputs: {[o.shape for o in sess.get_outputs()]}")
        
        return outputs
    except Exception as e:
        print(f"Error during inference ONNX model {model_path}: {e}")
        return None

def checking_inputs_outputs(model_path: str, providers: list = ["CUDAExecutionProvider"]):
    sess = ort.InferenceSession(model_path, providers=providers)

    print("== Inputs ==")
    for inp in sess.get_inputs():
        print(f"{inp.name}: shape={inp.shape}, dtype={inp.type}")

    print("\n== Outputs ==")
    for out in sess.get_outputs():
        print(f"{out.name}: shape={out.shape}, dtype={out.type}")



def export_onnx_models(prompt: str):
    with torch.no_grad():

        torch.onnx.export(
            vae,
            (torch.randn(1, 4, 64, 64).cuda(), False),
            os.path.join(OUTPUT_FOLDER, "vae.onnx"),
            input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "sample": {0: "batch"}
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
            args=(text_input.input_ids.to(device),),
            f=os.path.join(OUTPUT_FOLDER, "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
                "pooler_output": {0: "batch"}
            },
            
            opset_version=14,
            do_constant_folding=True,
        )
        
if __name__ == "__main__":
    
    # sess = ort.InferenceSession("/home/duong.quang.minh/project/Serving/model_repository/stable_diffusion_v1_4/text_encoder.onnx", providers=["CUDAExecutionProvider"])

    # text_input = tokenizer(
    #     "Draw a dog",
    #     return_tensors="np",
    #     padding="max_length",
    #     truncation=True,
    #     max_length=77
    # )
    
    # print("Text encoder input shape: ", text_input)
    # print("Text encoder input ids shape: ", text_input["input_ids"].shape)  # (1, 77)
    

    # outputs = sess.run(None, {"input_ids": text_input["input_ids"]})
    # print(outputs[0].shape, outputs[1].shape)  # last_hidden_state, pooler_output
    
    
    onnx_path = "/mnt/nvme1n1/duong.quang.minh/packtech_assets/triton/assets/ckpt/humanparsing/parsing_lip.onnx"
    
    inference_session(onnx_path, providers=["CUDAExecutionProvider"])