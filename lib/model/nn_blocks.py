#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py. """

import logging
from typing import Dict, Optional, Tuple, Union

from lib.utils import get_backend

from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, ReflectionPadding2D, Swish, KResizeImages
from .normalization import InstanceNormalization

if get_backend() == "amd":
    from keras.layers import (
        Activation, Add, Multiply, BatchNormalization, Concatenate, Conv2D as KConv2D, Conv2DTranspose,
        DepthwiseConv2D as KDepthwiseConv2d, LeakyReLU, PReLU, SeparableConv2D, UpSampling2D)
    from keras.initializers import he_uniform, VarianceScaling  # pylint:disable=no-name-in-module
    # type checking:
    import keras
    from plaidml.tile import Value as Tensor  # pylint:disable=import-error
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import (  # noqa pylint:disable=no-name-in-module,import-error
        Activation, Add, Multiply, BatchNormalization, Concatenate, Conv2D as KConv2D, Conv2DTranspose,
        DepthwiseConv2D as KDepthwiseConv2d, LeakyReLU, PReLU, SeparableConv2D, UpSampling2D)
    from tensorflow.keras.initializers import he_uniform, VarianceScaling  # noqa pylint:disable=no-name-in-module,import-error
    # type checking:
    from tensorflow import keras
    from tensorflow import Tensor
    import tensorflow as tf


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_CONFIG: dict = {}
_NAMES: Dict[str, int] = {}


def set_config(configuration: dict) -> None:
    """ Set the global configuration parameters from the user's config file.

    These options are used when creating layers for new models.

    Parameters
    ----------
    configuration: dict
        The configuration options that exist in the training configuration files that pertain
        specifically to Custom Faceswap Layers. The keys should be: `icnr_init`, `conv_aware_init`
        and 'reflect_padding'
     """
    global _CONFIG  # pylint:disable=global-statement
    _CONFIG = configuration
    logger.debug("Set NNBlock configuration to: %s", _CONFIG)


def _get_name(name: str) -> str:
    """ Return unique layer name for requested block.

    As blocks can be used multiple times, auto appends an integer to the end of the requested
    name to keep all block names unique

    Parameters
    ----------
    name: str
        The requested name for the layer

    Returns
    -------
    str
        The unique name for this layer
    """
    global _NAMES  # pylint:disable=global-statement,global-variable-not-assigned
    _NAMES[name] = _NAMES.setdefault(name, -1) + 1
    name = f"{name}_{_NAMES[name]}"
    logger.debug("Generating block name: %s", name)
    return name


#  << CONVOLUTIONS >>
def _get_default_initializer(
        initializer: keras.initializers.Initializer) -> keras.initializers.Initializer:
    """ Returns a default initializer of Convolutional Aware or he_uniform for convolutional
    layers.

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer` or None
        The initializer that has been passed into the model. If this value is ``None`` then a
        default initializer will be set to 'he_uniform'. If Convolutional Aware initialization
        has been enabled, then any passed through initializer will be replaced with the
        Convolutional Aware initializer.

    Returns
    -------
    :class:`keras.initializers.Initializer`
        The kernel initializer to use for this convolutional layer. Either the original given
        initializer, he_uniform or convolutional aware (if selected in config options)
    """
    if _CONFIG["conv_aware_init"]:
        retval = ConvolutionAware()
    elif initializer is None:
        retval = he_uniform()
    else:
        retval = initializer
        logger.debug("Using model supplied initializer: %s", retval)
    logger.debug("Set default kernel_initializer: (original: %s current: %s)", initializer, retval)

    return retval


class Conv2D(KConv2D):  # pylint:disable=too-few-public-methods, too-many-ancestors
    """ A standard Keras Convolution 2D layer with parameters updated to be more appropriate for
    Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to `he_uniform` or `convolutional
    aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    is_upscale: `bool`, optional
        ``True`` if the convolution is being called from an upscale layer. This causes the instance
        to check the user configuration options to see if ICNR initialization has been selected and
        should be applied. This should only be passed in as ``True`` from :class:`UpscaleBlock`
        layers. Default: ``False``
    """
    def __init__(self, *args, padding: str = "same", is_upscale: bool = False, **kwargs) -> None:
        if kwargs.get("name", None) is None:
            filters = kwargs["filters"] if "filters" in kwargs else args[0]
            kwargs["name"] = _get_name(f"conv2d_{filters}")
        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))
        if is_upscale and _CONFIG["icnr_init"]:
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        super().__init__(*args, padding=padding, kernel_initializer=initializer, **kwargs)


class DepthwiseConv2D(KDepthwiseConv2d):  # noqa,pylint:disable=too-few-public-methods, too-many-ancestors
    """ A standard Keras Depthwise Convolution 2D layer with parameters updated to be more
    appropriate for Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard
    :class:`keras.layers.DepthwiseConv2D` except where listed below. The default initializer is
    updated to `he_uniform` or `convolutional aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    is_upscale: `bool`, optional
        ``True`` if the convolution is being called from an upscale layer. This causes the instance
        to check the user configuration options to see if ICNR initialization has been selected and
        should be applied. This should only be passed in as ``True`` from :class:`UpscaleBlock`
        layers. Default: ``False``
    """
    def __init__(self, *args, padding: str = "same", is_upscale: bool = False, **kwargs) -> None:
        if kwargs.get("name", None) is None:
            kwargs["name"] = _get_name("dwconv2d")
        initializer = _get_default_initializer(kwargs.pop("depthwise_initializer", None))
        if is_upscale and _CONFIG["icnr_init"]:
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        super().__init__(*args, padding=padding, depthwise_initializer=initializer, **kwargs)


class Conv2DOutput():  # pylint:disable=too-few-public-methods
    """ A Convolution 2D layer that separates out the activation layer to explicitly set the data
    type on the activation to float 32 to fully support mixed precision training.

    The Convolution 2D layer uses default parameters to be more appropriate for Faceswap
    architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to he_uniform or convolutional
    aware based on user config settings.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int or tuple/list of 2 ints
        The height and width of the 2D convolution window. Can be a single integer to specify the
        same value for all spatial dimensions.
    activation: str, optional
        The activation function to apply to the output. Default: `"sigmoid"`
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int]],
                 activation: str = "sigmoid",
                 padding: str = "same", 
                 dtype: str = "float32", **kwargs) -> None:
        self._name = kwargs.pop("name") if "name" in kwargs else _get_name(
            f"conv_output_{filters}")
        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation
        self._padding = padding
        self._kwargs = kwargs
        self._dtype = dtype

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Convolutional Output Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Convolution 2D Layer
        """
        var_x = Conv2D(self._filters,
                       self._kernel_size,
                       padding=self._padding,
                       name=f"{self._name}_conv2d",
                       **self._kwargs)(inputs)
        var_x = Activation(self._activation, dtype=self._dtype, name=self._name)(var_x)
        return var_x


class Conv2DBlock():  # pylint:disable=too-few-public-methods
    """ A standard Convolution 2D layer which applies user specified configuration to the
    layer.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Adds instance normalization if requested. Adds a LeakyReLU if a residual block follows.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. NB: If `use_depthwise` is ``True`` then a value must still be provided here,
        but it will be ignored. Default: 5
    strides: tuple or int, optional
        An integer or tuple/list of 2 integers, specifying the strides of the convolution along the
        height and width. Can be a single integer to specify the same value for all spatial
        dimensions. Default: `2`
    padding: ["valid", "same"], optional
        The padding to use. NB: If reflect padding has been selected in the user configuration
        options, then this argument will be ignored in favor of reflect padding. Default: `"same"`
    normalization: str or ``None``, optional
        Normalization to apply after the Convolution Layer. Select one of "batch" or "instance".
        Set to ``None`` to not apply normalization. Default: ``None``
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    use_depthwise: bool, optional
        Set to ``True`` to use a Depthwise Convolution 2D layer rather than a standard Convolution
        2D layer. Default: ``False``
    relu_alpha: float
        The alpha to use for LeakyRelu Activation. Default=`0.1`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 5,
                 strides: Union[int, Tuple[int, int]] = 2,
                 padding: str = "same",
                 normalization: Optional[str] = None,
                 activation: Optional[str] = "leakyrelu",
                 use_depthwise: bool = False,
                 relu_alpha: float = 0.1,
                 **kwargs) -> None:
        self._name = kwargs.pop("name") if "name" in kwargs else _get_name(f"conv_{filters}")

        logger.debug("name: %s, filters: %s, kernel_size: %s, strides: %s, padding: %s, "
                     "normalization: %s, activation: %s, use_depthwise: %s, kwargs: %s)",
                     self._name, filters, kernel_size, strides, padding, normalization,
                     activation, use_depthwise, kwargs)

        self._use_reflect_padding = _CONFIG["reflect_padding"]

        self._args = (kernel_size, ) if use_depthwise else (filters, kernel_size)
        self._strides = strides
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs
        self._normalization = None if not normalization else normalization.lower()
        self._activation = None if not activation else activation.lower()
        self._use_depthwise = use_depthwise
        self._relu_alpha = relu_alpha

        self._assert_arguments()

    def _assert_arguments(self) -> None:
        """ Validate the given arguments. """
        assert self._normalization in ("batch", "instance", None), (
            "normalization should be 'batch', 'instance' or None")
        assert self._activation in ("leakyrelu", "swish", "prelu", None), (
            "activation should be 'leakyrelu', 'prelu', 'swish' or None")

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Convolution 2D Layer
        """
        if self._use_reflect_padding:
            inputs = ReflectionPadding2D(stride=self._strides,
                                         kernel_size=self._args[-1],
                                         name=f"{self._name}_reflectionpadding2d")(inputs)
        conv: keras.layers.Layer = DepthwiseConv2D if self._use_depthwise else Conv2D
        var_x = conv(*self._args,
                     strides=self._strides,
                     padding=self._padding,
                     name=f"{self._name}_{'dw' if self._use_depthwise else ''}conv2d",
                     **self._kwargs)(inputs)
        # normalization
        if self._normalization == "instance":
            var_x = InstanceNormalization(name=f"{self._name}_instancenorm")(var_x)
        if self._normalization == "batch":
            var_x = BatchNormalization(axis=3, name=f"{self._name}_batchnorm")(var_x)

        # activation
        if self._activation == "leakyrelu":
            var_x = LeakyReLU(self._relu_alpha, name=f"{self._name}_leakyrelu")(var_x)
        if self._activation == "swish":
            var_x = Swish(name=f"{self._name}_swish")(var_x)
        if self._activation == "prelu":
            var_x = PReLU(name=f"{self._name}_prelu")(var_x)

        return var_x


class SeparableConv2DBlock():  # pylint:disable=too-few-public-methods
    """ Seperable Convolution Block.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 5
    strides: tuple or int, optional
        An integer or tuple/list of 2 integers, specifying the strides of the convolution along
        the height and width. Can be a single integer to specify the same value for all spatial
        dimensions. Default: `2`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Separable
        Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 5,
                 strides: Union[int, Tuple[int, int]] = 2, **kwargs) -> None:
        self._name = _get_name(f"separableconv2d_{filters}")
        logger.debug("name: %s, filters: %s, kernel_size: %s, strides: %s, kwargs: %s)",
                     self._name, filters, kernel_size, strides, kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides

        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))
        kwargs["kernel_initializer"] = initializer
        self._kwargs = kwargs

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Separable Convolutional 2D Block.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = SeparableConv2D(self._filters,
                                kernel_size=self._kernel_size,
                                strides=self._strides,
                                padding="same",
                                name=f"{self._name}_seperableconv2d",
                                **self._kwargs)(inputs)
        var_x = Activation("relu", name=f"{self._name}_relu")(var_x)
        return var_x


#  << UPSCALING >>

class UpscaleBlock():  # pylint:disable=too-few-public-methods
    """ An upscale layer for sub-pixel up-scaling.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. NB: If reflect padding has been selected in the user configuration
        options, then this argument will be ignored in favor of reflect padding. Default: `"same"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    normalization: str or ``None``, optional
        Normalization to apply after the Convolution Layer. Select one of "batch" or "instance".
        Set to ``None`` to not apply normalization. Default: ``None``
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: str = "same",
                 scale_factor: int = 2,
                 normalization: Optional[str] = None,
                 activation: Optional[str] = "leakyrelu",
                 subpixel: Optional[int] = None,
                 **kwargs) -> None:
        self._name = _get_name(f"upscale_{filters}")
        logger.debug("name: %s. filters: %s, kernel_size: %s, padding: %s, scale_factor: %s, "
                     "normalization: %s, activation: %s, kwargs: %s)",
                     self._name, filters, kernel_size, padding, scale_factor, normalization,
                     activation, kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._scale_factor = scale_factor
        self._normalization = normalization
        self._activation = activation
        self._subpixel = subpixel or scale_factor
        self._kwargs = kwargs

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = Conv2DBlock(self._filters * self._scale_factor * self._scale_factor,
                            self._kernel_size,
                            strides=(1, 1),
                            padding=self._padding,
                            normalization=self._normalization,
                            activation=self._activation,
                            name=f"{self._name}_conv2d",
                            is_upscale=True,
                            **self._kwargs)(inputs)
        var_x = PixelShuffler(name=f"{self._name}_pixelshuffler",
                              size=(self._subpixel, self._subpixel))(var_x)
        return var_x


class Upscale2xBlock():  # pylint:disable=too-few-public-methods
    """ Custom hybrid upscale layer for sub-pixel up-scaling.

    Most of up-scaling is approximating lighting gradients which can be accurately achieved
    using linear fitting. This layer attempts to improve memory consumption by splitting
    with bilinear and convolutional layers so that the sub-pixel update will get details
    whilst the bilinear filter will get lighting.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    sr_ratio: float, optional
        The proportion of super resolution (pixel shuffler) filters to use. Non-fast mode only.
        Default: `0.5`
    fast: bool, optional
        Use a faster up-scaling method that may appear more rugged. Default: ``False``
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: str = "same",
                 activation: Optional[str] = "leakyrelu",
                 interpolation: str = "bilinear",
                 sr_ratio: float = 0.5,
                 scale_factor: int = 2,
                 fast: bool = False, **kwargs) -> None:
        self._name = _get_name(f"upscale2x_{filters}_{'fast' if fast else 'hyb'}")

        self._fast = fast
        self._filters = filters if self._fast else filters - int(filters * sr_ratio)
        self._kernel_size = kernel_size
        self._padding = padding
        self._interpolation = interpolation
        self._activation = activation
        self._scale_factor = scale_factor
        self._kwargs = kwargs

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Upscale 2x Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        if not self._fast:
            var_x_sr = UpscaleBlock(self._filters,
                                    kernel_size=self._kernel_size,
                                    padding=self._padding,
                                    scale_factor=self._scale_factor,
                                    activation=self._activation,
                                    **self._kwargs)(var_x)
        if self._fast or (not self._fast and self._filters > 0):
            var_x2 = Conv2D(self._filters, 3,
                            padding=self._padding,
                            is_upscale=True,
                            name=f"{self._name}_conv2d",
                            **self._kwargs)(var_x)
            var_x2 = UpSampling2D(size=(self._scale_factor, self._scale_factor),
                                  interpolation=self._interpolation,
                                  name=f"{self._name}_upsampling2D")(var_x2)
            if self._fast:
                var_x1 = UpscaleBlock(self._filters,
                                      kernel_size=self._kernel_size,
                                      padding=self._padding,
                                      scale_factor=self._scale_factor,
                                      activation=self._activation,
                                      **self._kwargs)(var_x)
                var_x = Add()([var_x2, var_x1])
            else:
                var_x = Concatenate(name=f"{self._name}_concatenate")([var_x_sr, var_x2])
        else:
            var_x = var_x_sr
        return var_x


class UpscaleResizeImagesBlock():  # pylint:disable=too-few-public-methods
    """ Upscale block that uses the Keras Backend function resize_images to perform the up scaling
    Similar in methodology to the :class:`Upscale2xBlock`

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: str = "same",
                 activation: Optional[str] = "leakyrelu",
                 scale_factor: int = 2,
                 interpolation: str = "bilinear") -> None:
        self._name = _get_name(f"upscale_ri_{filters}")
        self._interpolation = interpolation
        self._size = scale_factor
        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._activation = activation

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Resize Images Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs

        var_x_sr = KResizeImages(size=self._size,
                                 interpolation=self._interpolation,
                                 name=f"{self._name}_resize")(var_x)
        var_x_sr = Conv2D(self._filters, self._kernel_size,
                          strides=1,
                          padding=self._padding,
                          is_upscale=True,
                          name=f"{self._name}_conv")(var_x_sr)
        var_x_us = Conv2DTranspose(self._filters, 3,
                                   strides=2,
                                   padding=self._padding,
                                   name=f"{self._name}_convtrans")(var_x)
        var_x = Add()([var_x_sr, var_x_us])

        if self._activation == "leakyrelu":
            var_x = LeakyReLU(0.2, name=f"{self._name}_leakyrelu")(var_x)
        if self._activation == "swish":
            var_x = Swish(name=f"{self._name}_swish")(var_x)
        if self._activation == "prelu":
            var_x = PReLU(name=f"{self._name}_prelu")(var_x)
        return var_x


class UpscaleDNYBlock():  # pylint:disable=too-few-public-methods
    """ Upscale block that implements methodology similar to the Disney Research Paper using an
    upsampling2D block and 2 x convolutions

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    References
    ----------
    https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    size: int, optional
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D
        layers
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: str = "same",
                 activation: Optional[str] = "leakyrelu",
                 size: int = 2,
                 interpolation: str = "bilinear",
                 **kwargs) -> None:
        self._name = _get_name(f"upscale_dny_{filters}")
        self._interpolation = interpolation
        self._size = size
        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._activation = activation
        self._kwargs = kwargs

    def __call__(self, inputs: Tensor) -> Tensor:
        var_x = UpSampling2D(size=self._size,
                             interpolation=self._interpolation,
                             name=f"{self._name}_upsample2d")(inputs)
        for idx in range(2):
            var_x = Conv2DBlock(self._filters,
                                self._kernel_size,
                                strides=1,
                                padding=self._padding,
                                activation=self._activation,
                                relu_alpha=0.2,
                                name=f"{self._name}_conv2d_{idx + 1}",
                                is_upscale=True,
                                **self._kwargs)(var_x)
        return var_x

def pad_tensor(t, pattern):
    pattern = keras.backend.reshape(pattern, (1, 1, 1, -1))
    pattern2 = keras.backend.tile(pattern, (keras.backend.shape(t)[0], 1, t.shape[2], 1))
    t = Concatenate(axis=1)([pattern2, t])
    t = Concatenate(axis=1)([t, pattern2])
    pattern2 = keras.backend.tile(pattern, (keras.backend.shape(t)[0], t.shape[1], 1, 1))
    t = Concatenate(axis=2)([pattern2, t])
    t = Concatenate(axis=2)([t, pattern2])

    return t

alpha = 0.1
eps = 1e-5

def get_bn_bias(bn_layer):
    gamma, beta, mean, var = bn_layer.weights
    std = keras.backend.sqrt(var + eps)
    bn_bias = beta - mean * gamma / std

    return bn_bias

class RRRB():
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|


    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        self.expand_conv = Conv2D(ratio*n_feats, 1, padding="valid")
        self.fea_conv = Conv2D(ratio*n_feats, 3, padding="valid")
        self.reduce_conv = Conv2D(n_feats, 1, padding="valid")
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.expand_conv(x)
        out_identity = out
        
        # padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.weights[1]
        out = pad_tensor(out, b0)

        out = Add()([self.fea_conv(out), out_identity])
        out = self.reduce_conv(out)
        out = Add()([out, x])

        return out

class RRRBInference():
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Args:
        n_feats (int): The number of feature maps.
    """

    def __init__(self, n_feats):
        self.rep_conv = Conv2D(n_feats, 3, padding="same")
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.rep_conv(x)

        return out

class ERB():
    """ Enhanced residual block for building FEMN.

    Diagram:
        --RRRB--LeakyReLU--RRRB--
        
    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """

    def __init__(self, n_feats, ratio=2):
        self.n_feats = n_feats
        self.ratio = ratio
    
    def __call__(self, x: Tensor) -> Tensor:
        out = RRRB(self.n_feats, self.ratio)(x)
        out = LeakyReLU(alpha=alpha)(out)
        out = RRRB(self.n_feats, self.ratio)(out)

        return out

class ERBInference():
    """ Enhanced residual block for building FEMN.

    Diagram:
        --RRRB--LeakyReLU--RRRB--
        
    Args:
        n_feats (int): Number of feature maps.
    """

    def __init__(self, n_feats, ratio=0):
        self.n_feats = n_feats
    
    def __call__(self, x: Tensor) -> Tensor:
        out = RRRBInference(self.n_feats)(x)
        out = LeakyReLU(alpha=alpha)(out)
        out = RRRBInference(self.n_feats)(out)

        return out

class HFAB():
    """ High-Frequency Attention Block.

    Diagram:
        ---BN--Conv--[ERB]*up_blocks--BN--Conv--BN--Sigmoid--*--
         |___________________________________________________|

    Args:
        n_feats (int): Number of HFAB input feature maps.
        up_blocks (int): Number of ERBs for feature extraction in this HFAB.
        mid_feats (int): Number of feature maps in ERB.

    Note:
        Batch Normalization (BN) is adopted to introduce global contexts and achieve sigmoid unsaturated area.

    """

    def __init__(self, n_feats, up_blocks, mid_feats, ratio):
        self._name = _get_name("HFAB")
        self.n_feats = n_feats
        self.up_blocks = up_blocks
        self.mid_feats = mid_feats
        self.ratio = ratio
        self.bn1 = BatchNormalization(axis=3, epsilon=eps)
        self.bn2 = BatchNormalization(axis=3, epsilon=eps)
        self.bn3 = BatchNormalization(axis=3, epsilon=eps)

    def __call__(self, x: Tensor) -> Tensor:
        n_feats = self.n_feats
        mid_feats = self.mid_feats
        up_blocks = self.up_blocks
        ratio = self.ratio

        # padding with bn bias
        out = self.bn1(x)
        bn1_bias = get_bn_bias(self.bn1)
        out = pad_tensor(out, bn1_bias)

        # squeeze
        out = Conv2D(mid_feats, 3, padding="valid")(out)
        out = LeakyReLU(alpha=alpha)(out)

        for _ in range(up_blocks):
            out = ERB(mid_feats, ratio)(out)
        out = LeakyReLU(alpha=alpha)(out)

        # padding with bn bias
        out = self.bn2(out)
        bn2_bias = get_bn_bias(self.bn2)
        out = pad_tensor(out, bn2_bias)

        # excite
        out = Conv2D(n_feats, 3, padding="valid")(out)

        out = Activation("sigmoid", dtype="float32")(self.bn3(out))

        return Multiply()([out, x])


class HFABInference():
    """ High-Frequency Attention Block.

    Diagram:
        ---BN--Conv--[ERB]*up_blocks--BN--Conv--BN--Sigmoid--*--
         |___________________________________________________|

    Args:
        n_feats (int): Number of HFAB input feature maps.
        up_blocks (int): Number of ERBs for feature extraction in this HFAB.
        mid_feats (int): Number of feature maps in ERB.

    Note:
        Batch Normalization (BN) is adopted to introduce global contexts and achieve sigmoid unsaturated area.

    """

    def __init__(self, n_feats, up_blocks, mid_feats, ratio=0):
        self._name = _get_name("HFAB")
        self.n_feats = n_feats
        self.up_blocks = up_blocks
        self.mid_feats = mid_feats

    def __call__(self, x: Tensor) -> Tensor:
        n_feats = self.n_feats
        mid_feats = self.mid_feats
        up_blocks = self.up_blocks

        # squeeze
        out = Conv2D(mid_feats, 3, padding="same")(x)
        out = LeakyReLU(alpha=alpha)(out)

        for _ in range(up_blocks):
            out = ERBInference(mid_feats)(out)
        out = LeakyReLU(alpha=alpha)(out)

        # excite
        out = Conv2D(n_feats, 3, padding="same")(out)

        out = Activation("sigmoid", dtype="float32")(out)

        return Multiply()([out, x])


class FMEN():
    """ Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution

    Written by @aopsr
    
    https://arxiv.org/abs/2204.08397
    https://github.com/NJU-Jet/FMEN

    """

    def __init__(self, 
                in_ch: int = 3, 
                n_colors: int = 3,
                scale: int = 4,
                n_feats: int = 50,
                down_blocks: int = 4,
                inference: bool = True):
        self.down_blocks = down_blocks
        self.up_blocks = [2] + [1] * self.down_blocks
        self.n_feats = n_feats
        self.mid_feats = int(n_feats * 16 / 50) 
        self.backbone_expand_ratio = 2
        self.attention_expand_ratio = 2

        self.scale = [scale]
        self.n_colors = n_colors
        self.in_ch = in_ch
        self._name = _get_name("fmen")
        self.inference = inference
        self.hfab = HFABInference if inference else HFAB
        self.erb = ERBInference if inference else ERB

        if self.n_colors * (self.scale[0]**2) > self.n_feats:
            print("Warning: n_colors * (scale**2) > n_feats.")
        
        if self.n_colors * (self.scale[0]**2) < self.in_ch:
            print("Warning: n_colos * (scale**2) < in_ch")
    
    def __call__(self, x: Tensor) -> Tensor:
        up_blocks = self.up_blocks
        mid_feats = self.mid_feats
        n_feats = self.n_feats
        n_colors = self.n_colors
        scale = self.scale[0]
        backbone_expand_ratio = self.backbone_expand_ratio
        attention_expand_ratio = self.attention_expand_ratio

        # head
        x = Conv2D(n_feats, 3, padding="same", name="head")(x)

        # warm up
        h = Conv2D(n_feats, 3, padding="same", name="warmup_conv2d")(x)
        h = self.hfab(n_feats, up_blocks[0], mid_feats-4, attention_expand_ratio)(h)

        # body
        for i in range(self.down_blocks):
            h = self.erb(n_feats, backbone_expand_ratio)(h)
            h = self.hfab(n_feats, up_blocks[i+1], mid_feats, attention_expand_ratio)(h)

        h = Conv2D(n_feats, 3, padding="same", name="lr_conv")(h)

        h = Add()([h, x])

        # tail
        x = Conv2D(n_colors*(scale**2), 3, padding="same", name="tail")(h)
        x = PixelShuffler(name=f"{self._name}_pixelshuffler",
                              size=scale)(x)

        return x

# << OTHER BLOCKS >>
class ResidualBlock():  # pylint:disable=too-few-public-methods
    """ Residual block from dfaker.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer

    Returns
    -------
    tensor
        The output tensor from the Upscale layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: str = "same",
                 **kwargs) -> None:
        self._name = _get_name(f"residual_{filters}")
        logger.debug("name: %s, filters: %s, kernel_size: %s, padding: %s, kwargs: %s)",
                     self._name, filters, kernel_size, padding, kwargs)
        self._use_reflect_padding = _CONFIG["reflect_padding"]

        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the Faceswap Residual Block.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        if self._use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=self._kernel_size,
                                        name=f"{self._name}_reflectionpadding2d_0")(var_x)
        var_x = Conv2D(self._filters,
                       kernel_size=self._kernel_size,
                       padding=self._padding,
                       name=f"{self._name}_conv2d_0",
                       **self._kwargs)(var_x)
        var_x = LeakyReLU(alpha=0.2, name=f"{self._name}_leakyrelu_1")(var_x)
        if self._use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=self._kernel_size,
                                        name=f"{self._name}_reflectionpadding2d_1")(var_x)

        kwargs = {key: val for key, val in self._kwargs.items() if key != "kernel_initializer"}
        if not _CONFIG["conv_aware_init"]:
            kwargs["kernel_initializer"] = VarianceScaling(scale=0.2,
                                                           mode="fan_in",
                                                           distribution="uniform")
        var_x = Conv2D(self._filters,
                       kernel_size=self._kernel_size,
                       padding=self._padding,
                       name=f"{self._name}_conv2d_1",
                       **kwargs)(var_x)

        var_x = Add()([var_x, inputs])
        var_x = LeakyReLU(alpha=0.2, name=f"{self._name}_leakyrelu_3")(var_x)
        return var_x
