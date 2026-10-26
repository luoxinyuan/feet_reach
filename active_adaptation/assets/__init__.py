import os
import copy
from isaaclab.assets import Articulation
from .humanoid import *
from .sonic import G1_SONIC


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_col_full": G1_COL_FULL,
    "g1_sonic": G1_SONIC,
}
