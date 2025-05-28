import onnx
import torch
import argparse
import numpy as np
import onnxruntime as ort

from utils import setup_logging

logger = setup_logging()

# ONNX Checker function
def onnx_checker(mode_path: str = "models/vae.onnx"):
    try:
        onnx_model = onnx.load(mode_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        logger.error(f"Error when checking ONNX model: {e}")

# ONNX Inference function
def onnx_inference(model_path: str = "models/vae.onnx", **kwargs):
    try:
        ort_session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
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
        
        return outputs
    except Exception as e:
        logger.error(f"Error when inference ONNX model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on ONNX model")
    parser.add_argument("--model-path", type=str, default="models/vae.onnx", help="Path to ONNX model file")
    args = parser.parse_args()
        
    onnx_checker(args.model_path)
    
    # Inference
    dummy_input = torch.rand(1, 4, 64, 64)

    outputs = onnx_inference(
        model_path=args.model_path,
        latent_sample=dummy_input,
        # return_dict=np.array(False).astype(np.bool_)
    )
    print(type(outputs))
    print(outputs[0].shape) # 1, 3, 512, 512
    