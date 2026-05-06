"""
Hyperion Model Zoo
====================
Qlib-style 模型库，包含:

- LightGBM / XGBoost / CatBoost (GBDT)
- LSTM (长短期记忆网络)
- GRU (门控循环单元)
- GATs (Gated Attention Transformer)
- ALSTM (Attention LSTM)
- TabNet (表格注意力网络)

所有模型统一继承 BaseModel，通过 ModelRegistry 注册。
双后端: PyTorch + NumPy fallback.
"""

from hyperion.model_zoo.base import BaseModel, ModelRegistry

# GBDT models
from hyperion.model_zoo.gbdt import (
    LightGBMModel, LightGBMModelReg,
    XGBoostModel, XGBoostModelReg,
    CatBoostModel, CatBoostModelReg,
)

# LSTM
from hyperion.model_zoo.lstm import (
    LSTMModel, LSTMNumPyModel,
)

# GRU
from hyperion.model_zoo.gru import (
    GRUModel, GRUNumPyModel,
)

# GATs (Gated Attention Transformer)
from hyperion.model_zoo.gats import (
    GATsModel, GATsNumPyModel,
)

# ALSTM (Attention LSTM)
from hyperion.model_zoo.alstm import (
    ALSTMModel, ALSTMNumPyModel,
)

# TabNet
from hyperion.model_zoo.tabnet import (
    TabNetModel, TabNetNumPyModel,
)

__all__ = [
    "BaseModel", "ModelRegistry",
    "LightGBMModel", "LightGBMModelReg",
    "XGBoostModel", "XGBoostModelReg",
    "CatBoostModel", "CatBoostModelReg",
    "LSTMModel", "LSTMNumPyModel",
    "GRUModel", "GRUNumPyModel",
    "GATsModel", "GATsNumPyModel",
    "ALSTMModel", "ALSTMNumPyModel",
    "TabNetModel", "TabNetNumPyModel",
]