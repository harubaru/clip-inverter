import torch

# basically just AR CLIP prior from DALL-E 2 paper : https://arxiv.org/abs/2204.06125
class FeatureInverter(torch.nn.Module):
    def __init__(self, head = 4, enc_layers = 4, dec_layers = 4, dim_feedforward = 2048, input_feature_size: int = 257, input_feature_dim: int = 1024, output_feature_size: int = 77, output_feature_dim: int = 768, n_layers: int = 2, loss = 'l2') -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size

        if loss == 'l1':
          self.loss = torch.nn.functional.l1_loss
        elif loss == 'l2':
          self.loss = torch.nn.functional.mse_loss
        elif loss == 'huber':
          self.loss = torch.nn.functional.smooth_l1_loss

        self.in_proj = torch.nn.Linear(in_features=input_feature_dim, out_features=output_feature_dim)
        self.transformer = torch.nn.Transformer(d_model = output_feature_dim, nhead = head, num_encoder_layers = enc_layers, num_decoder_layers = dec_layers, dim_feedforward = dim_feedforward, batch_first = True)

    def forward(self, x: torch.tensor = None, y: torch.tensor = None, null_cond: torch.tensor = None, return_loss: bool = False) -> torch.tensor:
        assert x is not None
        if null_cond is None:
          null_cond = y
        x = self.in_proj(x) # 1024 -> 768
        outs = self.transformer(x, null_cond) # img -> txt
        if return_loss:
            return self.loss(outs, y)
        return outs
