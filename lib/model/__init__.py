#!/usr/bin/env python3
""" Conditional imports depending on whether the AMD version is installed or not """

from lib.utils import get_backend

from .normalization import (AdaInstanceNormalization, GroupNormalization,  # noqa
                            InstanceNormalization, LayerNormalization, RMSNormalization)
from .loss import losses  # noqa

if get_backend() == "amd":
    from . import optimizers_plaid as optimizers  # noqa
else:
    from . import optimizers_tf as optimizers  #type:ignore # noqa

from .efficientnetv2_lite import EfficientNetV2B0
from .efficientnetv2_lite import EfficientNetV2B1
from .efficientnetv2_lite import EfficientNetV2B2
from .efficientnetv2_lite import EfficientNetV2B3
from .efficientnetv2_lite import EfficientNetV2L
from .efficientnetv2_lite import EfficientNetV2M
from .efficientnetv2_lite import EfficientNetV2S