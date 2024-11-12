import torch


class MambaCache:
    def __init__(self, n_layers, batch_size=1, expand_factor=2, d_state=16, d_conv=4, dtype=torch.bfloat16, device='cuda'):
        factory_config = dict(device=device, dtype=dtype)
        self.cache = [(
            torch.zeros(batch_size, expand_factor * dim, d_state, **factory_config),
            torch.zeros(batch_size, expand_factor * dim, d_conv - 1, **factory_config)
        ) for _ in range(self.config.n_layers)]
