"""Map-free SMART decoder wrapper."""
from torch import nn
from src.model.agent_decoder import AirAgentDecoder


class AirDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.agent_decoder = AirAgentDecoder(**kwargs)

    def forward(self, tokenized, token_features, sampled=False):
        return self.agent_decoder(tokenized, token_features, sampled)
