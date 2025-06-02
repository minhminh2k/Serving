from typing import Dict

from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, ViTOnnxConfig

class CLIPVisionWithProjectionOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    DEFAULT_ONNX_OPSET = 14

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {
            "image_embeds": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs

class CLIPVisionOnnxConfig(CLIPVisionWithProjectionOnnxConfig):

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }


class CLIPTextModelWithProjectionOnnxConfig(CLIPTextOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_embeds": {0: "batch_size"},
        }
