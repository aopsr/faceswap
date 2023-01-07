#!/usr/bin/env python3
""" Xseg face mask plugin. """

import os
import sys

import numpy as np

from lib.model.session import KSession
from lib.utils import get_backend
from ._base import Masker, logger, BatchType, MaskerBatch

import pickle
from pathlib import Path
import cv2

if get_backend() == "amd":
    from keras.layers import (
        Add, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Input, Lambda, MaxPooling2D,
        ZeroPadding2D, Concatenate, Flatten, Reshape, Dense)
    from keras import backend as K
    from keras import initializers
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import (  # pylint:disable=no-name-in-module,import-error
        Add, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Input, Lambda, MaxPooling2D,
        ZeroPadding2D, Concatenate, Flatten, Reshape, Dense, Layer, Permute)
    from tensorflow.keras import backend as K
    from tensorflow.keras import initializers

class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs):
        model_filename = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "XSeg_256.npy")
        super().__init__(**kwargs)
        self.model_path = model_filename
        self.name = "Xseg"
        self.input_size = 256
        # to determine
        self.vram = 3714
        self.vram_warnings = 1088  # at BS 1. OOMs at higher batch sizes
        self.vram_per_batch = 114
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = Xseg(self.model_path,
                              allow_growth=self.config["allow_growth"],
                              exclude_gpus=self._exclude_gpus)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        batch.feed = np.array([feed.extract_face_xseg()
                                    for feed in batch.feed_faces],
                                   dtype="float32") / 255.0

        logger.trace("feed shape: %s", batch.feed.shape)

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        return self.model.predict(feed)[..., -1]

    def process_output(self, batch):
        """ Compile found faces for output """

        # convert dfl wf to fs face

        masks = []
        for prediction, feed in zip(batch.prediction, batch.feed_faces):
            prediction[prediction < 0.1] = 0
            
            matrix = feed.matrix.copy()
            padding = feed._padding_from_coverage(256, 1.0)["face"]
            matrix = matrix * (256 - 2 * padding)
            matrix[:, 2] += padding

            transform_matrix = np.dot(np.append(matrix, [[0,0,1]], axis=0), np.append(cv2.invertAffineTransform(feed._xseg_matrix), [[0,0,1]], axis=0))[0:2]
            mask = cv2.warpAffine(prediction, transform_matrix, (256,256), cv2.INTER_LANCZOS4)

            mask[mask<0.5] = 0
            mask[mask>=0.5] = 1
            masks.append(mask)

        batch.prediction = np.array(masks)
        return batch
    
class FRNorm2D(Layer):
    def __init__(self, in_ch, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.in_ch,),
            name="weight",
            initializer=initializers.Ones()
        )
        self.bias = self.add_weight(
            shape=(self.in_ch,),
            name="bias",
            initializer=initializers.Zeros()
        )
        self.eps = self.add_weight(
            shape=(1,),
            name="eps",
            initializer=initializers.Constant(1e-6)
        )
        self.built = True

    def call(self, x):
        shape = (1, 1, 1, self.in_ch)
        weight       = K.reshape ( self.weight, shape )
        bias         = K.reshape ( self.bias  , shape )
        nu2 = K.mean(K.square(x), axis=[1,2], keepdims=True)
        x = x * ( 1.0/K.sqrt(nu2 + K.abs(self.eps) ) )

        return x*weight + bias

class TLU(Layer):
    def __init__(self, in_ch, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch
        self.tau_initializer = initializers.Zeros()
    
    def build(self, input_shape):
        self.tau = self.add_weight(
            shape=(self.in_ch,),
            name="tau",
            initializer=self.tau_initializer
        )
        self.built = True

    def call(self, inputs):
        return K.maximum(inputs, self.tau)

class ConvBlock(Layer):
    def __init__(self, in_ch, out_ch, name):
        self.conv = Conv2D(out_ch, kernel_size=3, padding='same', name=name+"/conv")
        self.frn = FRNorm2D(out_ch, name=name+"/frn")
        self.tlu = TLU(out_ch, name=name+"/tlu")

    def __call__(self, x):                
        x = self.conv(x)
        x = self.frn(x)
        x = self.tlu(x)
        return x

class UpConvBlock(Layer):
    def __init__(self, in_ch, out_ch, name):
        self.conv = Conv2DTranspose(out_ch, kernel_size=3, strides=2, padding='same', name=name+"/conv")
        self.frn = FRNorm2D(out_ch, name=name+"/frn")
        self.tlu = TLU(out_ch, name=name+"/tlu")

    def __call__(self, x):
        x = self.conv(x)
        x = self.frn(x)
        x = self.tlu(x)
        return x

class BlurPool(Layer):
    def __init__(self, filt_size=3, stride=2, **kwargs ):

        self.strides = (stride,stride)

        self.filt_size = filt_size
        pad = [ int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ]

        self.padding = [ pad, pad ]

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:,None]*a[None,:]
        a = a / np.sum(a)
        a = a[:,:,None,None]
        self.k = K.constant(a, dtype=K.floatx())
        super().__init__(**kwargs)

    def __call__(self, x):
        k = K.tile (self.k, (1,1,x.shape[3],1) )
        x = K.spatial_2d_padding(x, padding=self.padding)
        x = K.depthwise_conv2d(x, k, strides=self.strides, padding='valid')
        return x

class Xseg(KSession):
    """ Xseg ported from DFL by @aopsr"""
    def __init__(self, model_path, allow_growth, exclude_gpus):
        super().__init__("Xseg",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus)

        self.define_model(self._model_definition)
            
        self.load_model_weights()
    
    def load_model_weights(self):
        filepath = Path(self._model_path)
        if filepath.exists():
            d_dumped = filepath.read_bytes()
            d = pickle.loads(d_dumped)
        else:
            raise Exception("xseg model not found")

        layers = self._model.layers

        try:
            for layer in layers: # TODO: reshape and permute dense layers to accommodate NCHW to NHWC
                if len(layer.trainable_weights):
                    names = [weight.name.replace("kernel", "weight") for weight in layer.trainable_weights]
                    weights = [d.get(name, None) for name in names]

                    layer.set_weights(weights)

        except:
            raise Exception("error loading xseg model weights")

    @classmethod
    def _model_definition(cls):
        """ Definition of Xseg.

        Returns
        -------
        tuple
            The tensor input to the model and tensor output to the model for compilation by
            :func`define_model`
        """
        input_ = Input(shape=(256, 256, 3))
        
        in_ch = 3
        base_ch = 32
        out_ch = 1

        conv01 = ConvBlock(in_ch, base_ch, name="conv01")
        conv02 = ConvBlock(base_ch, base_ch, name="conv02")
        bp0 = BlurPool (filt_size=4)

        conv11 = ConvBlock(base_ch, base_ch*2, name="conv11")
        conv12 = ConvBlock(base_ch*2, base_ch*2, name="conv12")
        bp1 = BlurPool (filt_size=3)

        conv21 = ConvBlock(base_ch*2, base_ch*4, name="conv21")
        conv22 = ConvBlock(base_ch*4, base_ch*4, name="conv22")
        bp2 = BlurPool (filt_size=2)

        conv31 = ConvBlock(base_ch*4, base_ch*8, name="conv31")
        conv32 = ConvBlock(base_ch*8, base_ch*8, name="conv32")
        conv33 = ConvBlock(base_ch*8, base_ch*8, name="conv33")
        bp3 = BlurPool (filt_size=2)

        conv41 = ConvBlock(base_ch*8, base_ch*8, name="conv41")
        conv42 = ConvBlock(base_ch*8, base_ch*8, name="conv42")
        conv43 = ConvBlock(base_ch*8, base_ch*8, name="conv43")
        bp4 = BlurPool (filt_size=2)
        
        conv51 = ConvBlock(base_ch*8, base_ch*8, name="conv51")
        conv52 = ConvBlock(base_ch*8, base_ch*8, name="conv52")
        conv53 = ConvBlock(base_ch*8, base_ch*8, name="conv53")
        bp5 = BlurPool (filt_size=2)

        dense1 = Dense(512, input_shape=(4*4*base_ch*8,), name="dense1")
        dense2 = Dense(4*4* base_ch*8, input_shape=(512,), name="dense2")

        up5 = UpConvBlock (base_ch*8, base_ch*4, name="up5")
        uconv53 = ConvBlock(base_ch*12, base_ch*8, name="uconv53")
        uconv52 = ConvBlock(base_ch*8, base_ch*8, name="uconv52")
        uconv51 = ConvBlock(base_ch*8, base_ch*8, name="uconv51")
        
        up4 = UpConvBlock (base_ch*8, base_ch*4, name="up4")
        uconv43 = ConvBlock(base_ch*12, base_ch*8, name="uconv43")
        uconv42 = ConvBlock(base_ch*8, base_ch*8, name="uconv42")
        uconv41 = ConvBlock(base_ch*8, base_ch*8, name="uconv41")

        up3 = UpConvBlock (base_ch*8, base_ch*4, name="up3")
        uconv33 = ConvBlock(base_ch*12, base_ch*8, name="uconv33")
        uconv32 = ConvBlock(base_ch*8, base_ch*8, name="uconv32")
        uconv31 = ConvBlock(base_ch*8, base_ch*8, name="uconv31")

        up2 = UpConvBlock (base_ch*8, base_ch*4, name="up2")
        uconv22 = ConvBlock(base_ch*8, base_ch*4, name="uconv22")
        uconv21 = ConvBlock(base_ch*4, base_ch*4, name="uconv21")

        up1 = UpConvBlock (base_ch*4, base_ch*2, name="up1")
        uconv12 = ConvBlock(base_ch*4, base_ch*2, name="uconv12")
        uconv11 = ConvBlock(base_ch*2, base_ch*2, name="uconv11")

        up0 = UpConvBlock (base_ch*2, base_ch, name="up0")
        uconv02 = ConvBlock(base_ch*2, base_ch, name="uconv02")
        uconv01 = ConvBlock(base_ch, base_ch, name="uconv01")
        out_conv = Conv2D(out_ch, kernel_size=3, padding='same', name="out_conv")

        concat = Concatenate(axis=3)

        x = conv01(input_)
        x = x0 = conv02(x)
        x = bp0(x)

        x = conv11(x)
        x = x1 = conv12(x)
        x = bp1(x)

        x = conv21(x)
        x = x2 = conv22(x)
        x = bp2(x)

        x = conv31(x)
        x = conv32(x)
        x = x3 = conv33(x)
        x = bp3(x)

        x = conv41(x)
        x = conv42(x)
        x = x4 = conv43(x)
        x = bp4(x)

        x = conv51(x)
        x = conv52(x)
        x = x5 = conv53(x)
        x = bp5(x)
        
        x = Permute((3, 1, 2))(x) # NHWC to NCHW
        x = Flatten()(x)
        x = dense1(x)
        x = dense2(x)
        x = Reshape((base_ch*8, 4, 4))(x)
        x = Permute((2, 3, 1))(x) #NCHW to NHWC

        x = up5(x)
        x = uconv53(concat([x, x5]))
        x = uconv52(x)
        x = uconv51(x)
        
        x = up4(x)
        x = uconv43(concat([x, x4]))
        x = uconv42(x)
        x = uconv41(x)

        x = up3(x)
        x = uconv33(concat([x, x3]))
        x = uconv32(x)
        x = uconv31(x)

        x = up2(x)
        x = uconv22(concat([x, x2]))
        x = uconv21(x)

        x = up1(x)
        x = uconv12(concat([x, x1]))
        x = uconv11(x)

        x = up0(x)
        x = uconv02(concat([x, x0]))
        x = uconv01(x)

        logits = out_conv(x)

        return input_, K.sigmoid(logits)