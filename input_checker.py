import os
import time
import onnx
import onnxruntime as ort
import argparse
import torch
import numpy as np
from PIL import Image
from src.shared.logging import setup_logging

logger = setup_logging()

SESS_OPTIONS = ort.SessionOptions()
SESS_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESS_OPTIONS.log_severity_level = 1

BASE_PATH = "/mnt/nvme1n1/duong.quang.minh/sdxl"
TEXT_ENCODER_1_PATH = os.path.join(BASE_PATH, "text_encoder", "model.onnx")
TEXT_ENCODER_2_PATH = os.path.join(BASE_PATH, "text_encoder_2", "model.onnx")
UNET_PATH = os.path.join(BASE_PATH, "unet", "model.onnx")
VAE_ENCODE_PATH = os.path.join(BASE_PATH, "vae_encoder", "model.onnx")
VAE_DECODE_PATH = os.path.join(BASE_PATH, "vae_decoder", "model.onnx")

# ONNX Checker function
def onnx_checker(mode_path: str = "models/vae.onnx"):
    try:
        onnx_model = onnx.load(mode_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        logger.error(f"Error when checking ONNX model: {e}")

def inference_onnx(model_path: str, **kwargs):
    try:
        ort_session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            sess_options=SESS_OPTIONS
        )
        
        for inp in ort_session.get_inputs():
            logger.info(f"Input name: {inp.name} --- Shape: {inp.shape} --- Type: {inp.type}")

        prepared_inputs = {
            k: (v.numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
            if isinstance(v, (np.ndarray, list, float, int)) or hasattr(v, "numpy")
        }

        outputs = ort_session.run(None, prepared_inputs)
        logger.info(f"Output names: {[o.name for o in ort_session.get_outputs()]}")
        logger.info(f"Outputs: {[o.shape for o in ort_session.get_outputs()]}")
        
        return outputs
    except Exception as e:
        logger.error(f"Error when inference ONNX model: {e}")
        return None

def get_inputs_outputs(model_path: str):
    try:
        model = onnx.load(model_path)
        inputs = {i.name: [d.dim_value if (d.dim_value > 0) else -1 for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
        outputs = {o.name: [d.dim_value if (d.dim_value > 0) else -1 for d in o.type.tensor_type.shape.dim] for o in model.graph.output}
        return inputs, outputs
    except Exception as e:
        logger.error(f"Error when loading ONNX model: {e}")
        return None, None
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Check ONNX model and perform inference")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the ONNX model file")
    args = parser.parse_args()

    inputs, outputs = get_inputs_outputs(args.onnx_model)
    print("Inputs:", inputs)
    print("Outputs:", outputs)
    
    # network = onnx.load(args.onnx_model)
    
    # for i in range(network.num_inputs):
    #     t = network.get_input(i)
    #     print(f"{t.name}: {t.shape}")

    
    # Inference example
    # text_encoder_inputs = {
    #     'input_ids': torch.randint(0, 1000, (1, 77), dtype=torch.int32),
    # }
    
    # unet_inputs = {
    #     'sample': torch.randn(1, 4, 64, 64, dtype=torch.float32),
    #     'timestep': torch.tensor([1000], dtype=torch.int64),
    #     'encoder_hidden_states': torch.randn(1, 77, 1024, dtype=torch.float32),
    #     'text_embeds': torch.randn(1, 77, 1024, dtype=torch.float32),
    #     'time_ids': torch.tensor([1, 6], dtype=torch.int64),
    # }
    
    # vae_encoder_inputs = {
    #     'sample': torch.randn(1, 3, 512, 512, dtype=torch.float16),
    # }
    
    # start_time = time.time()
    
    # model_path = "sdxl/vae_encoder/model.onnx"
    
    # s = inference_onnx(model_path, **vae_encoder_inputs)
    # print(s[0].shape)
    # end_time = time.time()
    # logger.info(f"Inference time: {end_time - start_time:.4f} seconds")
    