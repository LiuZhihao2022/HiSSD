REGISTRY = {}

from .sc2_decomposer import SC2Decomposer
from .sc2_decomposer_v2 import SC2DecomposerV2
REGISTRY["sc2_decomposer"] = SC2Decomposer
REGISTRY["sc2_decomposer_v2"] = SC2DecomposerV2

from .mpe_decomposer import MPEDecomposer # Add MPE decomposer import
REGISTRY["mpe_decomposer"] = MPEDecomposer # Register MPE decomposer

# from .gymma_decomposer import GYMMADecomposer

# REGISTRY["gymma_decomposer"] = GYMMADecomposer
