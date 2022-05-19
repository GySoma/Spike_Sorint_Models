import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

bce = keras.losses.MeanSquaredError()

def bce_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_pred[0])
    loss1 = bce(y_true, y_pred[1])
    loss2 = bce(y_true, y_pred[2])
    loss3 = bce(y_true, y_pred[3])

    return loss0 + loss1 + loss2 + loss3 

class ConvBlock(keras.layers.Layer):
    def __init__(self, out_ch=3,dirate=1):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate)
        self.bn = BatchNormalization()
        self.relu = ReLU()
    
    def call(self, inputs):
        hx = inputs

        x = self.conv(hx)
        x = self.bn(x)
        x = self.relu(x)

        return x

class RSU5(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1   = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2   = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)
        self.pool3   = MaxPool2D(2, strides=(2, 2))

        self.conv_b4 = ConvBlock(mid_ch, dirate=1)

        self.conv_b5 = ConvBlock(mid_ch, dirate=2)
    
    def call(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv_b4(hx)

        hx5 = self.conv_b5(hx4)
       
        return {'RSU5in' : hxin,
                'RSU5out': tf.concat([hx5, hx4], axis=3),
                'RSU5hx3': hx3,
                'RSU5hx2': hx2,
                'RSU5hx1': hx1,}

class RSU5D(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU5D, self).__init__()

        self.conv_b4_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
    
    def call(self, inputs):

        hxin = inputs["RSU5in"]
        hx4d = self.conv_b4_d(inputs["RSU5out"])
        hx4dup = self.upsample_3(hx4d)

        hx3d = self.conv_b3_d(tf.concat([hx4dup, inputs["RSU5hx3"]], axis=3))
        hx3dup = self.upsample_2(hx3d)

        hx2d =  self.conv_b2_d(tf.concat([hx3dup, inputs["RSU5hx2"]], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, inputs["RSU5hx1"]], axis=3))
        
        return hx1d + hxin

class RSU4(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1   = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2   = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)

        self.conv_b4 = ConvBlock(mid_ch, dirate=2)
    
    def call(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)

        hx4 = self.conv_b4(hx3)
        
        return {'RSU4in' : hxin,
                'RSU4out': tf.concat([hx4, hx3], axis=3),
                'RSU4hx2': hx2,
                'RSU4hx1': hx1,}

class RSU4D(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4D, self).__init__()

        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
    
    def call(self, inputs):

        hxin = inputs["RSU4in"]
        hx3d = self.conv_b3_d(inputs["RSU4out"])
        hx3dup = self.upsample_2(hx3d)

        hx2d =  self.conv_b2_d(tf.concat([hx3dup, inputs["RSU4hx2"]], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, inputs["RSU4hx1"]], axis=3))
        
        return hx1d + hxin


class RSU3(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU3, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1   = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)

        self.conv_b3 = ConvBlock(mid_ch, dirate=2)
    
    def call(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)

        hx3 = self.conv_b3(hx)
        
        return {'RSU3in' : hxin,
                'RSU3out': tf.concat([hx3, hx2], axis=3),
                'RSU3hx1': hx1,}

class RSU3D(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU3D, self).__init__()

        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
    
    def call(self, inputs):

        hxin = inputs["RSU3in"]

        hx2d = self.conv_b2_d(inputs["RSU3out"])
        hx2dup = self.upsample_1(hx2d)

        hx1d =  self.conv_b1_d(tf.concat([hx2dup, inputs["RSU3hx1"]], axis=3))

        return hx1d + hxin

class U2NETP(keras.models.Model):
    def __init__(self, out_ch=1):
        super(U2NETP, self).__init__()

        mid_chanel1 = 128
        mid_chanel2 = 128
        mid_chanel3 = 128
        out_chanel1 = 256
        out_chanel2 = 256
        out_chanel3 = 256

        self.stage1 = RSU5(mid_chanel1, out_chanel1)
        self.stage1D = RSU5D(mid_chanel1, out_chanel1)
        self.pool12 = MaxPool2D((2, 2), (2,2))

        self.stage2 = RSU4(mid_chanel2, out_chanel2)
        self.stage2D = RSU4D(mid_chanel2, out_chanel2)
        self.pool23 = MaxPool2D((2, 2), (2,2))

        self.stage3 = RSU3(mid_chanel3, out_chanel3)
        self.stage3D = RSU3D(mid_chanel3, out_chanel3)
        self.pool34 = MaxPool2D((2, 2), (2,2))

        self.stage3d = RSU3(mid_chanel3, out_chanel3)
        self.stage3dD = RSU3D(mid_chanel3, out_chanel3)
        self.side3 = Conv2D(out_ch, (3, 3), padding='same')

        self.stage2d = RSU4(mid_chanel2, out_chanel2)
        self.stage2dD = RSU4D(mid_chanel2, out_chanel2)
        self.side2 = Conv2D(out_ch, (3, 3), padding='same')

        self.stage1d = RSU5(mid_chanel1, out_chanel1)
        self.stage1dD = RSU5D(mid_chanel1, out_chanel1)
        self.side1 = Conv2D(out_ch, (3, 3), padding='same')

        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.upsample_out_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_out_3 = UpSampling2D(size=(4, 4), interpolation='bilinear')

        self.outconv = Conv2D(out_ch, (1, 1), padding='same')
    
    def call(self, inputs):
        hx = inputs

        hx1 = self.stage1(hx)
        hx1D = self.stage1D(hx1)
        hx = self.pool12(hx1D)

        hx2 = self.stage2(hx)
        hx2D = self.stage2D(hx2)
        hx = self.pool23(hx2D)

        hx3 = self.stage3(hx)
        hx3D = self.stage3D(hx3)
        hx = self.pool34(hx3D)

        hx3d = self.stage3d(hx3D)
        hx3dD = self.stage3dD(hx3d)
        hx3dup = self.upsample_3(hx3dD)
        side3 = self.upsample_out_3(self.side3(hx3dD))

        hx2d = self.stage2d(tf.concat([hx3dup, hx2D], axis=3))
        hx2dD = self.stage2dD(hx2d)
        hx2dup = self.upsample_2(hx2dD)
        side2 = self.upsample_out_2(self.side2(hx2dD))

        hx1d = self.stage1d(tf.concat([hx2dup, hx1D], axis=3))
        hx1dD = self.stage1dD(hx1d)
        side1 = self.side1(hx1dD)

        fused_output = self.outconv(tf.concat([side1, side2, side3], axis=3))

        return (tf.stack([(fused_output),  (side1), (side2), (side3)]),
                hx1["RSU5out"],
                hx1d["RSU5out"],
                hx2["RSU4out"],
                hx2d["RSU4out"],
                hx3["RSU3out"],
                hx3d["RSU3out"],
                )