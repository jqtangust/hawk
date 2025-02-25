import torch.nn as nn
import torch

class Projection(nn.Module):
    def __init__(self, llama_model):
        super(Projection, self).__init__()

        # Encoder
        self.encoder_0 = nn.Linear(
            llama_model.config.hidden_size, llama_model.config.hidden_size
        )
        self.encoder_1 = nn.Linear(
            llama_model.config.hidden_size, llama_model.config.hidden_size // 16
        )

        self.decoder_2 = nn.Linear(
            llama_model.config.hidden_size, llama_model.config.hidden_size
        )

    def forward(self, x):

        x_full = self.encoder_0(x)
        x_compress = self.encoder_1(x_full)

        x = self.decoder_2(x_full)

        return x, x_compress