#!/usr/bin/env python3
""" Xseg face mask plugin. """

import numpy as np

from lib.model.session import KSession
from lib.utils import get_backend
from ._base import Masker, logger

import pickle
from pathlib import Path

if get_backend() == "amd":
    from keras.layers import (
        Add, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Input, Lambda, MaxPooling2D,
        ZeroPadding2D, Concatenate, Flatten, Reshape, Dense)
    from keras import backend as K
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import (  # pylint:disable=no-name-in-module,import-error
        Add, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Input, Lambda, MaxPooling2D,
        ZeroPadding2D, Concatenate, Flatten, Reshape, Dense, Layer)
    from tensorflow.keras import backend as K
    import tensorflow_addons as tfa
    from tensorflow import keras
    from tensorflow import Variable
    import tensorflow as tf


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs):
        model_filename = "XSeg_256"
        super().__init__(**kwargs)
        self.model_path = model_filename
        self.name = "Xseg"
        self.input_size = 300
        self.vram = 2944
        self.vram_warnings = 1088  # at BS 1. OOMs at higher batch sizes
        self.vram_per_batch = 400
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = Xseg(self.model_path,
                              allow_growth=self.config["allow_growth"],
                              exclude_gpus=self._exclude_gpus)
        #self.model.append_softmax_activation(layer_index=-1)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        input_ = np.array([feed.face[..., :3]
                           for feed in batch["feed_faces"]], dtype="float32")
        batch["feed"] = input_ - np.mean(input_, axis=(1, 2))[:, None, None, :]
        logger.trace("feed shape: %s", batch["feed"].shape)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        predictions = self.model.predict(batch["feed"])
        batch["prediction"] = predictions[..., -1]
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch

class FRNorm2D(Layer):
    def __init__(self, in_ch):
        self.in_ch = in_ch
        super(FRNorm2D, self).__init__()
        self.weight = Variable(
            initial_value=keras.initializers.Ones()(shape=(in_ch,)), dtype="float32", trainable=True
        )
        self.bias = Variable(
            initial_value=keras.initializers.Zeros()(shape=(in_ch,)), dtype="float32", trainable=True
        )
        self.eps = Variable(
            initial_value=keras.initializers.Constant(1e-6)(shape=(1,)), dtype="float32", trainable=True
        )

    def __call__(self, x):
        shape = (1, 1, 1, self.in_ch)
        weight       = K.reshape ( self.weight, shape )
        bias         = K.reshape ( self.bias  , shape )
        nu2 = tf.math.reduce_mean(K.square(x), axis=[1,2], keepdims=True)
        x = x * ( 1.0/K.sqrt(nu2 + K.abs(self.eps) ) )

        return x*weight + bias

class ConvBlock(Layer):
    def __init__(self, in_ch, out_ch):              
        self.conv = Conv2D (out_ch, kernel_size=3, padding='same')
        self.frn = FRNorm2D(out_ch)
        self.tlu = tfa.layers.TLU()

    def __call__(self, x):                
        x = self.conv(x)
        x = self.frn(x)
        x = self.tlu(x)
        return x

class UpConvBlock(Layer):
    def __init__(self, in_ch, out_ch):
        self.conv = Conv2DTranspose (out_ch, kernel_size=3, padding='same')
        self.frn = FRNorm2D(out_ch)
        self.tlu = tfa.layers.TLU()

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
        k = K.tile (self.k, (1,1,x.shape[1],1) )
        x = K.spatial_2d_padding(x, padding=self.padding)
        x = K.depthwise_conv2d(x, k, strides=self.strides, padding='valid')
        return x

class Xseg(KSession):
    """ Xseg ported from DFL"""
    def __init__(self, model_path, allow_growth, exclude_gpus):
        super().__init__("Xseg",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus)

        self.define_model(self._model_definition)
        print(self._model.layers)
        self.load_model_weights()
    
    def load_model_weights(self):
        filepath = Path(self._model_path)
        if filepath.exists():
            d_dumped = filepath.read_bytes()
            d = pickle.loads(d_dumped)
        else:
            return False

        weights = self._model.layers

        try:
            arrays = []
            for w in weights:
                w_name_split = w.name.split('/')

                sub_w_name = "/".join(w_name_split[1:])

                w_val = d.get(sub_w_name, None)

                if w_val is None:
                    print("NOT LOADED")
                else:
                    w_val = np.reshape( w_val, w.shape.as_list() )
                    arrays.append(w_val)

            self._model.set_weight(arrays)
        except:
            return False

        return True
            

    @classmethod
    def _model_definition(cls):
        """ Definition of Xseg.

        Returns
        -------
        tuple
            The tensor input to the model and tensor output to the model for compilation by
            :func`define_model`
        """
        input_ = Input(shape=(300, 300, 3)) # need to change
        
        in_ch = 3
        base_ch = 32
        out_ch = 1

        conv01 = ConvBlock(in_ch, base_ch)
        conv02 = ConvBlock(base_ch, base_ch)
        bp0 = BlurPool (filt_size=4)

        conv11 = ConvBlock(base_ch, base_ch*2)
        conv12 = ConvBlock(base_ch*2, base_ch*2)
        bp1 = BlurPool (filt_size=3)

        conv21 = ConvBlock(base_ch*2, base_ch*4)
        conv22 = ConvBlock(base_ch*4, base_ch*4)
        bp2 = BlurPool (filt_size=2)

        conv31 = ConvBlock(base_ch*4, base_ch*8)
        conv32 = ConvBlock(base_ch*8, base_ch*8)
        conv33 = ConvBlock(base_ch*8, base_ch*8)
        bp3 = BlurPool (filt_size=2)

        conv41 = ConvBlock(base_ch*8, base_ch*8)
        conv42 = ConvBlock(base_ch*8, base_ch*8)
        conv43 = ConvBlock(base_ch*8, base_ch*8)
        bp4 = BlurPool (filt_size=2)
        
        conv51 = ConvBlock(base_ch*8, base_ch*8)
        conv52 = ConvBlock(base_ch*8, base_ch*8)
        conv53 = ConvBlock(base_ch*8, base_ch*8)
        bp5 = BlurPool (filt_size=2)

        dense1 = Dense(512, input_shape=(4*4*base_ch*8,))
        dense2 = Dense(4*4* base_ch*8, input_shape=(512,))

        up5 = UpConvBlock (base_ch*8, base_ch*4)
        uconv53 = ConvBlock(base_ch*12, base_ch*8)
        uconv52 = ConvBlock(base_ch*8, base_ch*8)
        uconv51 = ConvBlock(base_ch*8, base_ch*8)
        
        up4 = UpConvBlock (base_ch*8, base_ch*4)
        uconv43 = ConvBlock(base_ch*12, base_ch*8)
        uconv42 = ConvBlock(base_ch*8, base_ch*8)
        uconv41 = ConvBlock(base_ch*8, base_ch*8)

        up3 = UpConvBlock (base_ch*8, base_ch*4)
        uconv33 = ConvBlock(base_ch*12, base_ch*8)
        uconv32 = ConvBlock(base_ch*8, base_ch*8)
        uconv31 = ConvBlock(base_ch*8, base_ch*8)

        up2 = UpConvBlock (base_ch*8, base_ch*4)
        uconv22 = ConvBlock(base_ch*8, base_ch*4)
        uconv21 = ConvBlock(base_ch*4, base_ch*4)

        up1 = UpConvBlock (base_ch*4, base_ch*2)
        uconv12 = ConvBlock(base_ch*4, base_ch*2)
        uconv11 = ConvBlock(base_ch*2, base_ch*2)

        up0 = UpConvBlock (base_ch*2, base_ch)
        uconv02 = ConvBlock(base_ch*2, base_ch)
        uconv01 = ConvBlock(base_ch, base_ch)
        out_conv = Conv2D(out_ch, kernel_size=3, padding='same')

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
        
        x = Flatten()(x)
        x = dense1(x)
        x = dense2(x)
        x = Reshape(x, 4, 4, base_ch*8)(x)
                          
        x = up5(x)
        x = uconv53(Concatenate()([x, x5]))
        x = uconv52(x)
        x = uconv51(x)
        
        x = up4(x)
        x = uconv43(Concatenate()([x, x4]))
        x = uconv42(x)
        x = uconv41(x)

        x = up3(x)
        x = uconv33(Concatenate()([x, x3]))
        x = uconv32(x)
        x = uconv31(x)

        x = up2(x)
        x = uconv22(Concatenate()([x, x2]))
        x = uconv21(x)

        x = up1(x)
        x = uconv12(Concatenate()([x, x1]))
        x = uconv11(x)

        x = up0(x)
        x = uconv02(Concatenate()([x, x0]))
        x = uconv01(x)

        logits = out_conv(x)
        return input_, K.sigmoid(logits)
