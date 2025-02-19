import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt, pi, log2
from utils.masking import TriangularCausalMask, ProbMask
import os
from layers.Quatformer_EncDec import TrendNorm


class QuaternionAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super(QuaternionAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.query_size = query_size
        self.key_size = key_size

        query_pos = (torch.arange(0.0, query_size, 1.0) / query_size).view(-1, 1, 1)
        key_pos = (torch.arange(0.0, key_size, 1.0) / key_size).view(-1, 1, 1)
        self.register_buffer('query_pos', query_pos)
        self.register_buffer('key_pos', key_pos)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape  # B: batch, L: seq length, H: heads, E: embedding dim
        _, S, _, _ = keys.shape     # S: sequence length of keys
        _, _, _, M = query_omegas.shape  # M: number of learned frequencies

        # === Step 1: Compute Rotation Angles ===
        # Compute phase angles for the quaternion rotation using ω * pos + θ.
        # These angles are used in the exponential function e^(j(ω·pos + θ)).
        Q_angles = query_omegas * self.query_pos + query_thetas  # (B, L, H, M)
        K_angles = key_omegas * self.key_pos + key_thetas        # (B, S, H, M)

        # Compute cosine and sine values of the phase angles.
        # These represent the real and imaginary parts of e^(j * Q_angles).
        Q_cos, Q_sin = Q_angles.cos(), Q_angles.sin()  # (B, L, H, M)
        K_cos, K_sin = K_angles.cos(), K_angles.sin()  # (B, S, H, M)

        # === Step 2: Convert Queries/Keys to Quaternion Form ===
        # Split queries and keys into 4 components each, corresponding to a quaternion (q0, q1, q2, q3).
        # This is necessary to apply quaternion multiplication using the Hamilton product.
        Q_quaternion = torch.chunk(queries, 4, dim=-1)  # (B, L, H, E//4) * 4
        K_quaternion = torch.chunk(keys, 4, dim=-1)     # (B, S, H, E//4) * 4

        # === Step 3: Apply Quaternion Rotation ===
        # Using the quaternion multiplication rule, rotate Q and K by the computed angles.
        # This follows the Hamilton product, but specifically as a rotation using unit quaternions.
        Q_rotation = torch.cat(
            [
                # Applying quaternion rotation with cos/sin components
                torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_sin),
            ], dim=-1
        )  # (B, L, H, M, E//4)

        K_rotation = torch.cat(
            [
                torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_sin),
            ], dim=-1
        )  # (B, S, H, M, E//4)

        # === Step 4: Compute Attention Scores ===
        # Compute dot product between rotated queries and keys.
        # This corresponds to the series-attention with rotatory softmax-kernel.
        scale = self.scale or 1. / sqrt(E)  # Scaling factor for stability
        scores = torch.einsum("blhme,bshme->bhls", Q_rotation, K_rotation) / M  # (B, H, L, S)

        # === Step 5: Apply Softmax and Masking ===
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)  # Apply causal mask if needed

        # Compute final attention weights using softmax.
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # (B, H, L, S)

        # === Step 6: Compute Attention Output ===
        # Multiply attention weights with values (V) to get the final attended representations.
        V = torch.einsum("bhls,bshd->blhd", A, values)  # (B, L, H, E)

        # === Step 7: Return Outputs ===
        if self.output_attention:
            return V.contiguous(), A  # Return attention weights if required
        else:
            return V.contiguous(), None  # Otherwise, return only attention outputs

class FullAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
class LearningToRotateAttentionLayer(nn.Module):
    """
    A single Learning-to-Rotate Attention layer that:
      - Learns frequencies (omega) and phases (theta) for queries and keys
      - Applies a QuaternionAttention mechanism (self.inner_attention)
      - Computes regularization penalties for frequencies and phases
    """
    def __init__(self, attention, query_size, key_size, d_model, n_heads, period_type='variant', n_periods=2, d_keys=None,
                 d_values=None):

        super(LearningToRotateAttentionLayer, self).__init__()

        # Determine dimensionality of keys/values if not specified
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # The 'attention' argument is typically QuaternionAttention
        self.inner_attention = attention

        # Projections for queries, keys, and values into multi-head subspaces
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # First-order difference matrices for computing smoothness penalties on omega
        self.register_buffer('query_D_matrix', self._gen_D_matrix(query_size))
        self.register_buffer('key_D_matrix', self._gen_D_matrix(key_size))

        # Define layers to learn frequencies (omegas) and phases (thetas)
        # If period_type='variant', we use conv1d to learn time-varying frequencies.
        kernel_size = 1
        padding = kernel_size // 2
        if period_type == 'variant':
            self.query_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size,
                                                    padding=padding, padding_mode='zeros')
            self.key_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size,
                                                  padding=padding, padding_mode='zeros')
        else:
            # Otherwise, we learn a single set of frequencies for the entire sequence
            self.query_omega_projection = nn.Linear(d_model, n_periods * n_heads)
            self.key_omega_projection = nn.Linear(d_model, n_periods * n_heads)

        # Phases (thetas) always learned via conv1d, mapped into [-π, π] by tanh()
        self.query_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size,
                                                padding=padding, padding_mode='zeros')
        self.key_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size,
                                              padding=padding, padding_mode='zeros')

        # Output projection back to d_model dimension
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads
        self.period_type = period_type
        self.n_periods = n_periods

    def forward(self, queries, keys, values, attn_mask=None, is_training=False):
        """
        Forward pass for a single LRA layer:
          1) Compute (omega, theta) for queries/keys
          2) Apply QuaternionAttention
          3) Compute frequency & phase penalties
        """
        B, L, _ = queries.shape  # B: batch, L: length of queries
        _, S, _ = keys.shape     # S: length of keys
        H = self.n_heads         # number of attention heads

        # === 1) Generate frequencies (omegas) ===
        if self.period_type == 'variant':
            # Time-varying frequencies via conv1d for each position
            query_omegas = F.relu(self.query_omega_projection(queries.transpose(1, 2))) \
                               .transpose(1, 2).view(B, L, H, -1)
            key_omegas = F.relu(self.key_omega_projection(keys.transpose(1,2))) \
                             .transpose(1, 2).view(B, S, H, -1)
        else:
            # Fixed frequencies across the sequence (using the mean embedding)
            query_omegas = F.relu(self.query_omega_projection(torch.mean(queries, dim=1))) \
                               .view(B, 1, H, -1).repeat(1, L, 1, 1)
            key_omegas = F.relu(self.key_omega_projection(torch.mean(keys, dim=1))) \
                             .view(B, 1, H, -1).repeat(1, S, 1, 1)
    
        # === 2) Generate phases (thetas) in range [-π, π] ===
        # We use tanh(...) * pi to scale outputs into [-π, π].
        query_thetas = (F.tanh(self.query_theta_projection(queries.transpose(1, 2)).transpose(1, 2)) * pi) \
                          .view(B, L, H, -1)
        key_thetas = (F.tanh(self.key_theta_projection(keys.transpose(1, 2)).transpose(1, 2)) * pi) \
                        .view(B, S, H, -1)

        # Project queries, keys, values for multi-head attention
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # === 3) Apply QuaternionAttention (rotatory softmax-kernel) ===
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            query_omegas,
            query_thetas,
            key_omegas,
            key_thetas,
            attn_mask
        )
        out = out.view(B, L, -1)

        # === 4) Compute frequency & phase penalties ===
        # This corresponds to Eq. (3) in your figure: the difference-based penalty for omega,
        # and the L2 norm-based penalty for theta.
        #   L_omega ~ sum_{n=0}^{N-2} (omega_{p}^{(n+1)} - omega_{p}^{(n)})^2
        #   L_theta ~ sum_{n=0}^{N-1} (theta_{p}^{(n)})^2
        query_omegas_diff = torch.einsum('ji,bihm->bjhm', self.query_D_matrix, query_omegas)
        key_omegas_diff = torch.einsum('ji,bihm->bjhm', self.key_D_matrix, key_omegas)

        query_omegas_penalty = torch.sum(query_omegas_diff ** 2)
        key_omegas_penalty = torch.sum(key_omegas_diff ** 2)
        query_thetas_penalty = torch.sum(query_thetas ** 2)
        key_thetas_penalty = torch.sum(key_thetas ** 2)

        omegas_penalty = (query_omegas_penalty + key_omegas_penalty)
        thetas_penalty = (query_thetas_penalty + key_thetas_penalty)

        # Return outputs + attention + penalty terms
        return self.out_projection(out), attn, omegas_penalty, thetas_penalty

    def _gen_D_matrix(self, L):
        """
        Create a first-order difference matrix for a sequence of length L.
        This is used to compute (omega[n+1] - omega[n]) for smoothness regularization.
        """
        D = torch.zeros(L - 1, L)
        D[:, 1:] = torch.eye(L - 1)
        D[:, :-1] -= torch.eye(L - 1)
        return D

    def _init(self, tensor):
        """
        Utility init function: uniform initialization for a given tensor.
        """
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor


class DecouplingLearningtoRotateAttentionLayer(nn.Module):
    """
    A 'decoupled' version of Learning-to-Rotate Attention that:
      - Uses two LRA layers with an intermediate 'inducing' representation
      - First, an LRA layer processes the (inducings, keys, values)
      - Then, a second LRA layer processes (queries, updated_inducings, updated_inducings)
      - This can help handle longer sequences or reduce memory usage by decoupling the attention steps.
    """
    def __init__(self, query_size, key_size, d_model, n_heads, period_type='variant', n_periods=2, d_keys=None,
                 d_values=None, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(DecouplingLearningtoRotateAttentionLayer, self).__init__()

        # We define an 'inducing' sequence of fixed size
        self.inducing_size = 96

        # Register a buffer for the inducing states, initialized randomly
        self.register_buffer('I', self._init(torch.zeros(1, self.inducing_size, d_model)))

        # Create two QuaternionAttention modules for two-step attention
        attn_1 = QuaternionAttention(self.inducing_size, key_size, mask_flag, scale, attention_dropout, output_attention)
        attn_2 = QuaternionAttention(query_size, self.inducing_size, mask_flag, scale, attention_dropout, output_attention)

        # An optional normalization step for the inducing sequence
        self.norm = TrendNorm(d_model, self.inducing_size, kernel_size=25)

        # Two LRA layers: each is a single LearningToRotateAttentionLayer
        # 1) LRA from (inducings -> keys/values)
        # 2) LRA from (queries -> updated_inducings)
        self.attn_layer_1 = LearningToRotateAttentionLayer(attn_1, self.inducing_size, key_size, d_model, n_heads)
        self.attn_layer_2 = LearningToRotateAttentionLayer(attn_2, query_size, self.inducing_size, d_model, n_heads)

    def forward(self, queries, keys, values, attn_mask=None, is_training=False):
        """
        Decoupled LRA forward pass:
          1) Inducing states attend to (keys, values) with the first LRA layer
          2) Normalize the updated inducing states
          3) Optionally update the stored self.I with a slow-moving average
          4) Queries attend to the updated inducing states with the second LRA layer
          5) Return the final output + combined penalty terms
        """
        # === 1) Expand the inducing states for the current batch
        inducings = self.I.repeat(queries.size(0), 1, 1)

        # First LRA layer: (inducings, keys, values)
        inducings, _, omegas_penalty_1, thetas_penalty_1 = self.attn_layer_1(inducings, keys, values)

        # Normalize the updated inducing states (optional step)
        inducings = self.norm(inducings)

        # If in training mode, we slowly update the stored self.I with the new average
        if is_training:
            I_new = inducings.detach().mean(0, keepdim=True)
            self.I = (1 - 1e-4) * self.I + 1e-4 * I_new

        # Second LRA layer: (queries, inducings, inducings)
        out, _, omegas_penalty_2, thetas_penalty_2 = self.attn_layer_2(queries, inducings, inducings)

        # Combine penalty terms from both LRA layers
        omegas_penalty = omegas_penalty_1 + omegas_penalty_2
        thetas_penalty = thetas_penalty_1 + thetas_penalty_2

        # Return final output + no attention matrix + total penalties
        return out, None, omegas_penalty, thetas_penalty

    def _init(self, tensor):
        """
        Utility init function: uniform initialization for a given tensor.
        """
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor
    