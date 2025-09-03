import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vq.residual_vq import ResidualVQ
from .vq.vector_quantize_pytorch import lens_to_mask, VectorQuantize


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, seq, hidden)

        quantization pipeline:

            1. get encoder input (B,S,H)
            2. flatten input to (B*S,H)

        """
        z_flattened = z.view(-1, self.e_dim)
        mask_expanded = mask.unsqueeze(-1)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean(((z_q.detach() - z) ** 2) * mask_expanded) + self.beta * \
            torch.mean(((z_q - z.detach()) ** 2) * mask_expanded)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.contiguous()

        quantized_results = {
            'quantized_feats': z_q,
            'quantized_indices': min_encoding_indices,
            'commit_loss': loss,
        }
        return quantized_results


class RVQAudioQuantizer(nn.Module):
    def __init__(
        self,
        dim = 1280, # input_dim
        num_quantizers = 4, # residual depth
        codebook_dim = None, # codebook dim, whether to down-project from dim --> codebook_dim or not, default will set to dim
        quantize_dropout = False, # set to true to allow raw input flowing through
        # rvq kwargs related to vq
        kmeans_init = True, 
        codebook_size = 256,
        decay = 0.99, 
        **vq_kwargs,
    ):  
        super().__init__()
        self.rvq = ResidualVQ(
            dim = dim, 
            num_quantizers = num_quantizers,
            codebook_dim = codebook_dim,
            quantize_dropout = quantize_dropout,
            # vq-related
            kmeans_init = kmeans_init,
            codebook_size = codebook_size,
            decay = decay,
            **vq_kwargs,
        )
    
    def forward(
        self,
        z,
        mask,
        **kwargs,
    ):
        quantized, indices, commit_loss = self.rvq(z, mask=mask)
        # print(commit_loss) # should be with shape: (num_quantizers,)
        sum_commit_loss_rvq = commit_loss.sum() # sum over the loss for the total one

        quantized_results = {
            'quantized_feats': quantized,
            'quantized_indices': indices,
            'commit_loss': sum_commit_loss_rvq,
        }
        return quantized_results


class VQAudioQuantizer(nn.Module):
    def __init__(
        self,
        **vq_kwargs,
    ):  
        super().__init__()
        self.vq = VectorQuantize(
            **vq_kwargs,
        )
    
    def forward(
        self,
        z,
        mask,
        **kwargs,
    ):
        quantized, indices, commit_loss = self.vq(z, mask=mask)
        # print(commit_loss) # should be with shape: (num_quantizers,)
        sum_commit_loss_vq = commit_loss.sum() # sum over the loss for the total one

        quantized_results = {
            'quantized_feats': quantized,
            'quantized_indices': indices,
            'commit_loss': sum_commit_loss_vq,
        }
        return quantized_results


class KmeansAudioQuantizer(nn.Module):
    def __init__(
        self,
        **vq_kwargs,
    ):  
        super().__init__()
        kmeans_path = '/media/ycc/results/kmeans_ws/5000/kmeans-98-38_765.pt'
        codebook = torch.load(kmeans_path, map_location='cpu')
        self.register_buffer('codebook', codebook)

    def pairwise_distance(self, z, codebook):
        # z: [B L M]
        # [B L 1 M]
        A = z.unsqueeze(dim=2)

        # [1 1 N M]
        B = codebook.unsqueeze(dim=0).unsqueeze(dim=0)

        distance = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        distance = distance.sum(dim=-1)
        return distance  # [B L N]

    def forward(
        self,
        z,
        mask,
        **kwargs,
    ):
        print('here')
        codebook = self.codebook.detach()
        distance = self.pairwise_distance(z, codebook)
        indices = torch.argmin(distance, dim=-1) # [B L]

        B, L = indices.shape
        N, C = codebook.shape

        # Reshape indexes to be a 1D tensor
        flat_indices = indices.reshape(-1)

        # Use index_select to get the codes
        codes = torch.index_select(codebook, dim=0, index=flat_indices)

        # Reshape the result to (B, L, C)
        quantized = codes.reshape(B, L, C)

        quantized[~mask] = 0.

        quantized_results = {
            'quantized_feats': quantized,
            'quantized_indices': indices,
            'commit_loss': 0.0,
        }
        return quantized_results


class NoAudioQuantizer(nn.Module):
    def __init__(
        self,
        dim = 1280, # input_dim
        codebook_dim = 256,
        kmeans_pt = None,
    ):  
        super().__init__()
        self.proj_in = nn.Linear(dim, codebook_dim)
        self.proj_out = nn.Linear(codebook_dim, dim)
        self.use_kmeans = False
        if kmeans_pt:
            codebook = torch.load(kmeans_pt, map_location='cpu')
            self.register_buffer('codebook', codebook)
            self.use_kmeans = True
            

    def pairwise_distance(self, z, codebook):
        # z: [B L M]
        # [B L 1 M]
        A = z.unsqueeze(dim=2)

        # [1 1 N M]
        B = codebook.unsqueeze(dim=0).unsqueeze(dim=0)

        distance = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        distance = distance.sum(dim=-1)
        return distance  # [B L N]

    def forward(
        self,
        z,
        mask,
        **kwargs,
    ):
        intermediate_hiddens = self.proj_in(z)
        z = intermediate_hiddens
        if self.use_kmeans:
            codebook = self.codebook.detach()
            distance = self.pairwise_distance(z, codebook)
            indices = torch.argmin(distance, dim=-1) # [B L]

            B, L = indices.shape
            N, C = codebook.shape

            # Reshape indexes to be a 1D tensor
            flat_indices = indices.reshape(-1)

            # Use index_select to get the codes
            codes = torch.index_select(codebook, dim=0, index=flat_indices)

            # Reshape the result to (B, L, C)
            z = codes.reshape(B, L, C)

        z = self.proj_out(z)
        z[~mask] = 0.

        quantized_results = {
            'quantized_feats': z,
            'quantized_indices': indices if self.use_kmeans else None,
            'commit_loss': 0.0,
            'intermediate_hiddens': intermediate_hiddens.detach()
        }
        return quantized_results


QUANTIZER_CLASSES = {
    'rvq': RVQAudioQuantizer,
    'vq': VQAudioQuantizer,
    'kmeans': KmeansAudioQuantizer,
    'no': NoAudioQuantizer,
}
