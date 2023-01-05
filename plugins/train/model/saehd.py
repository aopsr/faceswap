import numpy as np

from lib.model.nn_blocks import (
    Conv2DBlock, Conv2DOutput, ResidualBlock, UpscaleBlock )
from lib.model.normalization import LayerNormalization
from lib.utils import get_backend

from ._base import KerasModel, ModelBase
if get_backend() == "amd":
    from keras import backend as K
    from keras.layers import (
         Concatenate, Dense, Flatten, Input, Reshape )
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras import backend as K  # pylint:disable=import-error
    from tensorflow.keras.layers import (  # pylint:disable=import-error,no-name-in-module
        Concatenate, Dense, Flatten, Input, Reshape )

class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = self.config["output_size"]
        self.ae_dims = self.config["ae_dims"]
        self.e_dims = self.config["e_dims"]
        self.d_dims = self.config["d_dims"]
        self.learn_mask = self.config["learn_mask"]

        self.archi_type = self.config["archi_type"]
        self.u = self.config["u"]
        self.d = self.config["d"]
        self.t = self.config["t"]

        self.resolution = np.clip(((self.resolution // 32) * 32 if self.d else (self.resolution // 16) * 16), 64, 640)
        self.ae_dims = np.clip(self.ae_dims, 32, 1024)
        self.e_dims = np.clip(self.e_dims, 16, 256)
        self.e_dims = self.e_dims + self.e_dims % 2
        self.d_dims = np.clip(self.d_dims, 16, 256)
        self.d_dims = self.d_dims + self.d_dims % 2
        
        opts = ""
        if self.u:
            opts += "u"
        if self.d:
            opts += "d"
        if self.t:
            opts += "t"
        if opts:
            opts = "-" + opts

        self.model_type = f"{self.archi_type}{opts} {self.resolution}"

        self.encoder_res = self.resolution // 2 if self.d else self.resolution
        self.input_shape = (self.encoder_res, self.encoder_res, 3)

        self.lowest_dense_res = self.resolution // (32 if self.d else 16)
    
    def build_model(self, inputs):
        # swap to be consistent with DFL
        input_a = inputs[1] # src
        input_b = inputs[0] # dst

        encoder = self.encoder(self.e_dims)
        encoder_a = [encoder(input_a)]
        encoder_b = [encoder(input_b)]

        inter_input_shape = K.int_shape(encoder_a[0])[1:]

        if self.archi_type == "df":
            inter = self.inter(self.ae_dims, "inter", inter_input_shape)
            inter_a, inter_b = inter(encoder_a), inter(encoder_b)
            decoder_input_shape = K.int_shape(inter_a)[1:]
            outputs = [self.decoder(self.d_dims, "b", decoder_input_shape)(inter_b), 
                       self.decoder(self.d_dims, "a", decoder_input_shape)(inter_a)]
        else:
            inter_ab = self.inter(self.ae_dims, "inter_ab", inter_input_shape)

            inter_ab_src = inter_ab(encoder_a)
            inter_ab_src = Concatenate()([inter_ab_src, inter_ab_src])

            inter_b_dst = self.inter(self.ae_dims, "inter_b", inter_input_shape)(encoder_b)
            inter_ab_dst = inter_ab(encoder_b)
            inter_ab_dst = Concatenate()([inter_b_dst, inter_ab_dst])
            decoder_input_shape = K.int_shape(inter_ab_src)[1:]

            decoder = self.decoder(self.d_dims, "both", decoder_input_shape)
            outputs = [decoder(inter_ab_dst), decoder(inter_ab_src)]

        autoencoder = KerasModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def encoder(self, e_ch):
        input_ = Input(shape=self.input_shape)
        var_x = input_

        if self.t:
            var_x = Conv2DBlock(e_ch, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(e_ch)(var_x)
            var_x = Conv2DBlock(e_ch*2, activation="leakyrelu")(var_x)
            var_x = Conv2DBlock(e_ch*4, activation="leakyrelu")(var_x)
            var_x = Conv2DBlock(e_ch*8, activation="leakyrelu")(var_x)
            var_x = Conv2DBlock(e_ch*8, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(e_ch*8)(var_x)
        else:
            for i in range(4):
                var_x = Conv2DBlock(e_ch*(min(2**i, 8)), activation="leakyrelu")(var_x)
        
        if self.u:
            var_x = LayerNormalization()(var_x)
            
        var_x = Flatten()(var_x)

        # if self.u:
        #     pixel norm

        return KerasModel(input_, var_x, name="encoder")
    
    def inter(self, ae_ch, name, input_shape):
        input_ = Input(shape=input_shape)
        var_x = input_

        var_x = Dense(ae_ch)(var_x)
        var_x = Dense(self.lowest_dense_res ** 2 * ae_ch * 2)(var_x)
        var_x = Reshape((self.lowest_dense_res, self.lowest_dense_res, ae_ch * 2))(var_x)
        if not self.t:
            var_x = UpscaleBlock(ae_ch*2, activation="leakyrelu")(var_x)
        
        return KerasModel(input_, var_x, name=name)
    
    def decoder(self, d_ch, name, input_shape):
        input_ = Input(shape=input_shape)
        var_x = input_
        
        if self.t:
            var_x = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*8)(var_x)
            var_x = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*8)(var_x)
            var_x = UpscaleBlock(d_ch*4, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*4)(var_x)
            var_x = UpscaleBlock(d_ch*2, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*2)(var_x)

            if self.learn_mask:
                var_y = input_
                var_y = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_y)
                var_y = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_y)
                var_y = UpscaleBlock(d_ch*4, activation="leakyrelu")(var_y)
                var_y = UpscaleBlock(d_ch*2, activation="leakyrelu")(var_y)

        else:
            var_x = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*8)(var_x)
            var_x = UpscaleBlock(d_ch*4, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*4)(var_x)
            var_x = UpscaleBlock(d_ch*2, activation="leakyrelu")(var_x)
            var_x = ResidualBlock(d_ch*2)(var_x)

            if self.learn_mask:
                var_y = input_
                var_y = UpscaleBlock(d_ch*8, activation="leakyrelu")(var_y)
                var_y = UpscaleBlock(d_ch*4, activation="leakyrelu")(var_y)
                var_y = UpscaleBlock(d_ch*2, activation="leakyrelu")(var_y)
        
        if self.d:
            var_x = UpscaleBlock(d_ch*2, activation="leakyrelu")(var_x)
            
            if self.learn_mask:
                var_y = UpscaleBlock(d_ch, activation="leakyrelu")(var_y)
        
        var_x = Conv2DOutput(filters = 3, kernel_size = 1, 
                            name="face_out_" + name, dtype="float32")(var_x)
        
        outputs = [var_x]

        if self.learn_mask:
            var_y = Conv2DOutput(filters = 1, kernel_size = 1, 
                            name="mask_out_" + name, dtype="float32")(var_y)
            outputs.append(var_y)

        return KerasModel(input_, outputs=outputs, name=name)
