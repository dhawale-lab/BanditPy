from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DecisionModel
    from .regression_models import Logistic2Arm
    from .rnn.memory_ann import MemoryANNFit2Arm, MemoryANNModel
    from .rnn.rnn_fit import VanillaRNNFit2Arm, VanillaRNNModel
    from .rnn.rnn_models import BanditTrainer2Arm

__all__ = [
    "BanditTrainer2Arm",
    "DecisionModel",
    "Logistic2Arm",
    "MemoryANNFit2Arm",
    "MemoryANNModel",
    "VanillaRNNFit2Arm",
    "VanillaRNNModel",
]


def __getattr__(name):
    if name == "Logistic2Arm":
        from .regression_models import Logistic2Arm

        return Logistic2Arm
    if name == "BanditTrainer2Arm":
        from .rnn.rnn_models import BanditTrainer2Arm

        return BanditTrainer2Arm
    if name == "DecisionModel":
        from .model import DecisionModel

        return DecisionModel
    if name == "VanillaRNNModel":
        from .rnn.rnn_fit import VanillaRNNModel

        return VanillaRNNModel
    if name == "VanillaRNNFit2Arm":
        from .rnn.rnn_fit import VanillaRNNFit2Arm

        return VanillaRNNFit2Arm
    if name == "MemoryANNModel":
        from .rnn.memory_ann import MemoryANNModel

        return MemoryANNModel
    if name == "MemoryANNFit2Arm":
        from .rnn.memory_ann import MemoryANNFit2Arm

        return MemoryANNFit2Arm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
