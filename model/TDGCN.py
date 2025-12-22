import torch
import torch.nn as nn
from function.graph_conv import calculate_laplacian_with_self_loop
from config import cfg
from tensorly.decomposition import tucker


device = cfg['device']

class TDGCN(nn.Module):
    def __init__(
            self,cfg, **kwargs
    ):
        super(TDGCN, self).__init__()
        self._input_dim = cfg['batchsize']
        self._classnumber = cfg['ClassNumber']
        self._Datalen = cfg['DataLen']
        self.regressor = (
            nn.Linear(
                self._input_dim,
                self._classnumber,
            )
        )
        self.weights_Adaptive = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._input_dim)
        )
        self.weights_Adaptive_2 = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._input_dim)
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._Datalen, self._input_dim)
        )
        self.weights_projection = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._Datalen)
        )
        self.weights_2 = nn.Parameter(
            torch.FloatTensor((self._input_dim), self._input_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.weights_2)
        nn.init.xavier_uniform_(self.weights_Adaptive)
        nn.init.xavier_uniform_(self.weights_projection)    #  /
        nn.init.xavier_uniform_(self.weights_Adaptive_2)    #  /

    def forward(self, inputs):
        inputs = inputs.squeeze()
        num_components, _ = inputs.shape
        inputs_numpy = inputs.cpu().numpy()
        core, factors = tucker(inputs_numpy, rank=[num_components, num_components])
        core[core < 0] = 0
        core_tensor = torch.from_numpy(core).to(device)
        laplacian=calculate_laplacian_with_self_loop(core_tensor)
        laplacian_Adaptive = laplacian * self.weights_Adaptive
        ax = laplacian_Adaptive @ inputs
        H1 = ax @ torch.tanh(self.weights) + laplacian_Adaptive
        H2 = laplacian_Adaptive @ H1 * self.weights_2
        output = self.regressor(H2)
        return output

