REGISTRY = {}

from .mlp_agent import MLPAgent
from .rnn_agent import RNNAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent
from .deepset_agent import DeepsetAgent
from .hpn_agent import HPNActor
from .dimenet_agent import DimenetAgent
from .deepset_agent_mlp import DeepsetMLP
from .rdhnet_agent import RDHAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["deepset"] = DeepsetAgent
REGISTRY["hpn"] = HPNActor
REGISTRY["dimenet"] = DimenetAgent
REGISTRY["deepset_mlp"] = DeepsetMLP
REGISTRY["rdhnet"] = RDHAgent