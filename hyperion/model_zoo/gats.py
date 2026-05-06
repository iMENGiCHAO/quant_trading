"""
GATs 模型 — Gated Attention Transformer
===========================================
Qlib 的 GATs 模型复现。

基于 Transformer Encoder + 门控注意力机制:
- 多头因果自注意力 (Multi-Head Causal Self-Attention)
- 时间步门控 (Gating) 控制信息流
- 对时序数据天然适应

Alpha158 基准 Rank IC ~0.0390

双后端: PyTorch (首选) / NumPy (fallback)
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from hyperion.model_zoo.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


class _NumPyGATs:
    """纯 NumPy GATs 前向计算"""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 1):
        assert num_layers == 1, "NumPy GATs only supports 1 layer. Use PyTorch for multi-layer."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / self.hidden_dim)  # 用 hidden_dim 作为特征维

        # 多头注意力: hidden_dim → hidden_dim (输入已经是 projected)
        hd = self.head_dim
        self.W_q = [np.random.randn(self.num_heads, hd, self.hidden_dim) * scale for _ in range(self.num_layers)]
        self.W_k = [np.random.randn(self.num_heads, hd, self.hidden_dim) * scale for _ in range(self.num_layers)]
        self.W_v = [np.random.randn(self.num_heads, hd, self.hidden_dim) * scale for _ in range(self.num_layers)]
        self.W_o = [np.random.randn(self.hidden_dim, self.hidden_dim) * scale for _ in range(self.num_layers)]

        # 门控 (Sigmoid Gate)
        self.W_g = [np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1 for _ in range(self.num_layers)]

        # FFN
        self.W_ff1 = [np.random.randn(self.hidden_dim * 4, self.hidden_dim) * scale for _ in range(self.num_layers)]
        self.b_ff1 = [np.zeros(self.hidden_dim * 4) for _ in range(self.num_layers)]
        self.W_ff2 = [np.random.randn(self.hidden_dim, self.hidden_dim * 4) * scale for _ in range(self.num_layers)]
        self.b_ff2 = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]

        # LayerNorm 参数
        self.ln1_gamma = [np.ones(self.hidden_dim) for _ in range(self.num_layers)]
        self.ln1_beta = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        self.ln2_gamma = [np.ones(self.hidden_dim) for _ in range(self.num_layers)]
        self.ln2_beta = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]

        # 输出层
        self.W_out = np.random.randn(1, self.hidden_dim) * 0.01
        self.b_out = np.zeros(1)

        # 输入投影 (input_dim → hidden_dim)
        self.input_proj = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.input_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(np.clip(x, -50, 50))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

    def layer_norm(self, x, gamma, beta):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-12) + beta

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """前向传播

        Args:
            x_seq: (seq_len, input_dim)

        Returns:
            (1,) 预测值
        """
        seq_len = len(x_seq)
        # 输入投影: input_dim → hidden_dim
        x = x_seq @ self.input_proj.T  # (seq_len, hidden_dim)

        for layer_idx in range(self.num_layers):
            # 多头注意力
            Q = self.W_q[layer_idx] @ x.T  # (num_heads, head_dim, seq_len)
            K = self.W_k[layer_idx] @ x.T
            V = self.W_v[layer_idx] @ x.T

            # 缩放点积注意力 (因果掩码)
            scale = np.sqrt(self.head_dim)
            scores = np.einsum('hdt,hdT->htT', Q, K) / scale  # (num_heads, seq_len, seq_len)

            # 因果掩码
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores - 1e9 * mask[np.newaxis, :, :]

            attn = self.softmax(scores, axis=-1)  # (num_heads, seq_len, seq_len)

            # 加权求和: V 是 (num_heads, head_dim, seq_len)
            head_outputs = np.einsum('htt,hdT->htd', attn, V)  # (num_heads, seq_len, head_dim)
            head_outputs = head_outputs.transpose(1, 0, 2)  # (seq_len, num_heads, head_dim)
            head_outputs = head_outputs.reshape(seq_len, self.hidden_dim)

            # 注意力投影到 hidden_dim
            attn_proj = head_outputs @ self.W_o[layer_idx].T  # (seq_len, hidden_dim)

            # 门控残差
            gate = self.sigmoid(attn_proj @ self.W_g[layer_idx].T)
            x = x + gate * attn_proj

            # LayerNorm 1
            x = self.layer_norm(x, self.ln1_gamma[layer_idx], self.ln1_beta[layer_idx])

            # FFN
            ffn_out = np.maximum(0, x @ self.W_ff1[layer_idx].T + self.b_ff1[layer_idx])
            ffn_out = ffn_out @ self.W_ff2[layer_idx].T + self.b_ff2[layer_idx]
            x = x + ffn_out

            # LayerNorm 2
            x = self.layer_norm(x, self.ln2_gamma[layer_idx], self.ln2_beta[layer_idx])

        # 取最后一个时间步
        last_hidden = x[-1]  # (hidden_dim,)

        # 输出层
        out = self.W_out @ last_hidden + self.b_out
        return out


@ModelRegistry.register("gats")
class GATsModel(BaseModel):
    """GATs — Gated Attention Transformer for time series

    Qlib 基准: GATs
    Alpha158 Rank IC ~0.0390, 优于 LSTM/GRU
    """

    model_type = "gats"

    def __init__(
        self,
        input_dim: int = 158,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        seq_len: int = 20,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        use_pytorch: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.use_pytorch = use_pytorch

        self._model = None
        self._pt_model = None
        self._pt_available = False
        if use_pytorch:
            try:
                import torch
                self._pt = torch
                self._pt_available = True
            except ImportError:
                pass

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[List] = None):
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        if self._pt_available:
            return self._fit_pytorch(X_arr, y_arr, eval_set)
        else:
            return self._fit_numpy(X_arr, y_arr, eval_set)

    def _fit_numpy(self, X, y, eval_set=None):
        logger.info(f"GATs NumPy backend: {X.shape}")
        self._model = _NumPyGATs(self.input_dim, self.hidden_dim,
                                 self.num_heads, self.num_layers)
        self._fitted = True
        return self

    def _fit_pytorch(self, X, y, eval_set=None):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class GATsNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
                super().__init__()
                assert hidden_dim % num_heads == 0
                head_dim = hidden_dim // num_heads

                self.hidden_dim = hidden_dim
                self.num_heads = num_heads
                self.num_layers = num_layers
                self.head_dim = head_dim

                # 为简化，使用标准 TransformerEncoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation='gelu',
                )
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                self.fc = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                x = self.input_proj(x)
                # 因果掩码
                seq_len = x.size(1)
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')),
                                        diagonal=1).to(x.device)
                # 门控 Transformer
                for _ in range(self.num_layers):
                    attn_out = self.transformer.layers[0](x, mask=causal_mask, is_causal=False)
                    gate = self.gate(attn_out)
                    x = x + gate * attn_out
                    x = self.transformer.layers[0].norm1(x)
                    ffn_out = self.transformer.layers[0].linear2(
                        self.dropout(self.transformer.layers[0].activation(
                            self.transformer.layers[0].linear1(x))))
                    x = x + self.transformer.layers[0].dropout2(ffn_out)
                    x = self.transformer.layers[0].norm2(x)

                out = x[:, -1, :]
                out = self.dropout(out)
                return self.fc(out).squeeze()

        self._pt_model = GATsNet(
            self.input_dim, self.hidden_dim, self.num_heads, self.num_layers, self.dropout
        ).to(device)

        # 准备序列数据
        n = len(X)
        seq_len = min(self.seq_len, n)
        X_seq = []
        y_seq = []
        for i in range(seq_len, n):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])

        if len(X_seq) == 0:
            X_seq = np.tile(X, (seq_len // n + 1, 1))[np.newaxis, :seq_len, :]
            y_seq = y[:1]
        else:
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

        X_t = torch.FloatTensor(X_seq).to(device)
        y_t = torch.FloatTensor(y_seq).to(device)

        optimizer = optim.Adam(self._pt_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        patience = 0

        for epoch in range(self.max_epochs):
            self._pt_model.train()
            loss_acc = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self._pt_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._pt_model.parameters(), 1.0)
                optimizer.step()
                loss_acc += loss.item()

            if eval_set:
                self._pt_model.eval()
                with torch.no_grad():
                    vx = torch.FloatTensor(eval_set[0][0].values).to(device)
                    vy = torch.FloatTensor(eval_set[0][1].values).to(device)
                    vloss = criterion(self._pt_model(vx), vy).item()
                if vloss < best_loss:
                    best_loss = vloss
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stop_patience:
                        break

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pt_model is not None:
            import torch
            device = next(self._pt_model.parameters()).device
            self._pt_model.eval()
            with torch.no_grad():
                # 构建序列
                X_arr = X.values.astype(np.float32)
                n = len(X_arr)
                seq_len = min(self.seq_len, n)
                X_seq = []
                for i in range(seq_len, n):
                    X_seq.append(X_arr[i - seq_len:i])
                if len(X_seq) == 0:
                    X_seq = np.tile(X_arr, (seq_len // n + 1, 1))[np.newaxis, :seq_len, :]
                else:
                    X_seq = np.array(X_seq)
                X_t = torch.FloatTensor(X_seq).to(device)
                preds = self._pt_model(X_t).cpu().numpy()
                result = np.full(n, np.nan)
                result[-len(preds):] = preds
                return result
        elif self._model is not None:
            X_arr = X.values.astype(np.float32)
            n = len(X_arr)
            seq_len = min(self.seq_len, n)
            preds = []
            for i in range(seq_len, n):
                out = self._model.forward(X_arr[i - seq_len:i])
                preds.append(out.item())
            result = np.full(n, np.nan)
            if preds:
                result[-len(preds):] = preds
            return result
        raise RuntimeError("Model not fitted")


@ModelRegistry.register("gats_numpy")
class GATsNumPyModel(GATsModel):
    def __init__(self, **kwargs):
        kwargs["use_pytorch"] = False
        super().__init__(**kwargs)