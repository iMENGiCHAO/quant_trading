"""
ALSTM 模型 — Attention LSTM
==============================
Qlib 的 ALSTM 基准模型复现。

LSTM + 加性注意力机制 (Additive Attention) 的组合。

Alpha158 基准 Rank IC ~0.0363

双后端: PyTorch (首选) / NumPy (fallback)
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from hyperion.model_zoo.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


class _NumPyALSTM:
    """纯 NumPy ALSTM 前向计算 (LSTM + Attention)"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / self.input_dim)

        # LSTM 权重
        self.W_i = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.U_i = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b_i = np.zeros((self.hidden_dim, 1))
        self.W_f = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.U_f = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b_f = np.ones((self.hidden_dim, 1))
        self.W_o = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.U_o = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b_o = np.zeros((self.hidden_dim, 1))
        self.W_c = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.U_c = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b_c = np.zeros((self.hidden_dim, 1))

        # Attention 权重 (加性注意力)
        self.W_att = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.v_att = np.random.randn(self.hidden_dim, 1) * 0.1
        self.b_att = np.zeros((1, 1))

        # 输出层
        self.W_out = np.random.randn(1, self.hidden_dim) * 0.01
        self.b_out = np.zeros(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(np.clip(x, -50, 50))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """前向传播

        Args:
            x_seq: (seq_len, input_dim)

        Returns:
            (1,) 预测值
        """
        seq_len = x_seq.shape[0]
        h = np.zeros((self.num_layers, self.hidden_dim))
        c = np.zeros((self.num_layers, self.hidden_dim))

        # 存储所有时间步的隐含状态用于注意力
        all_hidden = []

        for t in range(seq_len):
            x_t = x_seq[t, :, np.newaxis]  # (input_dim, 1)

            for layer in range(self.num_layers):
                h_prev = h[layer, :, np.newaxis]
                c_prev = c[layer, :, np.newaxis]

                x_eff = x_t if layer == 0 else h[layer - 1, :, np.newaxis]

                i = self.sigmoid(self.W_i @ x_eff + self.U_i @ h_prev + self.b_i)
                f = self.sigmoid(self.W_f @ x_eff + self.U_f @ h_prev + self.b_f)
                o = self.sigmoid(self.W_o @ x_eff + self.U_o @ h_prev + self.b_o)
                c_tilde = self.tanh(self.W_c @ x_eff + self.U_c @ h_prev + self.b_c)

                c[layer] = (f * c_prev + i * c_tilde).flatten()
                h[layer] = (o * self.tanh(c[layer, :, np.newaxis])).flatten()

            all_hidden.append(h[-1].copy())

        # 加性注意力
        H = np.array(all_hidden)  # (seq_len, hidden_dim)
        scores = H @ self.W_att @ self.v_att + self.b_att  # (seq_len, 1)
        attn_weights = self.softmax(scores, axis=0)  # (seq_len, 1)

        # 注意力加权和
        context = (attn_weights * H).sum(axis=0)  # (hidden_dim,)

        # 输出
        out = self.W_out @ context + self.b_out
        return out


@ModelRegistry.register("alstm")
class ALSTMModel(BaseModel):
    """ALSTM — Attention LSTM 时序预测模型

    Qlib 基准: ALSTM
    Alpha158 Rank IC ~0.0363
    """

    model_type = "alstm"

    def __init__(
        self,
        input_dim: int = 158,
        hidden_dim: int = 64,
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
        logger.info(f"ALSTM NumPy backend: {X.shape}")
        self._model = _NumPyALSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self._fitted = True
        return self

    def _fit_pytorch(self, X, y, eval_set=None):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class ALSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # 注意力权重
                attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
                attn_weights = torch.softmax(attn_scores, dim=1)
                # 注意力上下文
                context = (attn_weights * lstm_out).sum(dim=1)
                context = self.dropout(context)
                return self.fc(context).squeeze()

        self._pt_model = ALSTMNet(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        ).to(device)

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
                preds = self._pt_model(X_t).cpu().numpy().flatten()
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


@ModelRegistry.register("alstm_numpy")
class ALSTMNumPyModel(ALSTMModel):
    def __init__(self, **kwargs):
        kwargs["use_pytorch"] = False
        super().__init__(**kwargs)