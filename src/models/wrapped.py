import torch
from transformers import CLIPTextModel

class WrappedTextEncoder(torch.nn.Module):
    def __init__(self, model: CLIPTextModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(
            input_ids=input_ids, 
            return_dict=True,
            output_hidden_states=True,
        )
        
        last_hidden_state = out.last_hidden_state
        pooler_output = out.pooler_output
        final_hidden_state = out.hidden_states[-2] # Test

        # Dummy Onnx graph
        final_hidden_state = final_hidden_state + 0
        pooler_output = pooler_output + 0
        last_hidden_state = last_hidden_state + 0

        return last_hidden_state, pooler_output, final_hidden_state