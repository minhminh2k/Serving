from optimum.exporters.onnx import main_export
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path

model_id = "openai/clip-vit-large-patch14"
output_dir = Path("onnx_clip_text_encoder")

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    opset=14,
    device="cuda",
    fp16=True,
)


# optimum-cli export onnx --model openai/clip-vit-large-patch14 \
#     --task feature-extraction \
#     --opset 14 \
#     --fp16 \
#     --device cuda \
#     --output onnx_clip_fp16
