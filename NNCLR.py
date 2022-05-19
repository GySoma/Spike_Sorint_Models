import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from NNCLR_data_generator import *
import numpy as np
import tensorflow.python.autograph as autograph

AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 5000
# The below two values are taken from https://www.tensorflow.org/datasets/catalog/stl10
labelled_train_images = 5000
unlabelled_images = 100000

temperature = 0.1
queue_size = 10000
random_sigma = tf.random.uniform(shape=(),minval=0.5,maxval=1.5)
random_noise_level = tf.random.uniform(shape=(),minval=1,maxval=2)
contrastive_augmenter = {
    "noise":random_noise_level,
    "sigma": random_sigma,
    "contrast_limit_lower":2,
    "contrast_limit_upper":2.5,
    "name": "contrastive_augmenter",
}
classification_augmenter = {
    "noise":random_noise_level,
    "sigma": random_sigma,
    "contrast_limit_lower":2,
    "contrast_limit_upper":2.5,
    "name": "classification_augmenter",
}
input_shape = (128, 64, 1)
width = 75
num_epochs = 100
num_steps = 5000
steps_per_epoch = 200
batch_size = 32

"""
### Augmentations
"""

class RandomNoise(layers.Layer):
    def __init__(self, noise_level):
        super(RandomNoise, self).__init__()
        self.noise_level = noise_level

    def random_noise(self, images):
        noise = np.random.normal(0,self.noise_level,8192)
        noise = np.reshape(noise, (128,64, 1))
        noise = tf.convert_to_tensor(noise,dtype=tf.float32)
        noise_images = images + noise
        return noise_images

    def call(self, images):
        images = self.random_noise(images)
        return images

class GaussBlur(layers.Layer):
    def __init__(self, blur_level):
        super(GaussBlur, self).__init__()
        self.blur_level = blur_level

    def gaussian_kernel(self, size: int,
                    mean: float,
                    std: float,
                   ):
        """Makes 2D gaussian Kernel for convolution."""

        d = tfp.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij',
                                    vals,
                                    vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)
    
    def call(self, images):
        gauss_kernel = self.gaussian_kernel(3,0,self.blur_level)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        images =tf.nn.conv2d(images, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        return images

class Random_Contrast(layers.Layer):
    def __init__(self, contrast_limit_lower, contrast_limit_upper):
        super(Random_Contrast, self).__init__()
        self.contrast_limit_lower = contrast_limit_lower
        self.contrast_limit_upper = contrast_limit_upper

    def random_contrast(self, images):
        return tf.image.random_contrast(images,self.contrast_limit_lower,self.contrast_limit_upper)

    def call(self, images):
        images = self.random_contrast(images)
        return images

"""
### Prepare augmentation module
"""


def augmenter(name,noise,sigma,contrast_limit_lower,contrast_limit_upper,brightness):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            RandomNoise(noise_level=noise),
            GaussBlur(blur_level=sigma),
            Random_Contrast(contrast_limit_lower=contrast_limit_lower,contrast_limit_upper=contrast_limit_upper),
        ],
        name=name,
    )

@autograph.convert()
def filter_negatives(x,y):

    negs_x = x[y==0]
    negs_y = y[y==0]

    x = x[y!=0]
    y = y[y!=0]

    x = tf.concat([x, negs_x[:len(x)//5]], axis=0)
    y = tf.concat([y, negs_y[:len(y)//5]], axis=0)
    return x,y

def res_2(x, s=(1,1), filter=64): 

    x_skip = x
    x = layers.Conv2D(filter, kernel_size=(3, 3), strides=s, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)

    x = layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x_skip = layers.Conv2D(filter, kernel_size=(1, 1), strides=s, padding='same')(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    x = layers.Add()([x, x_skip])
    x = layers.Activation(keras.activations.relu)(x)

    return x

def encoder():
    inp = tf.keras.layers.Input(shape=input_shape)
    x = inp

    #ResNet_12
    x = layers.BatchNormalization()(x)
    x = res_2(x,s=(2,2),filter=(64))
    x = res_2(x,s=(1,1),filter=(64))
    x = res_2(x,s=(2,2),filter=(128))
    x = res_2(x,s=(1,1),filter=(128))
    x = res_2(x,s=(2,2),filter=(256))
    x = res_2(x,s=(1,1),filter=(256))

    x = tf.keras.layers.Conv2D(
      filters=width,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="same",
      )(x)
    x = tf.keras.layers.Reshape((-1, width))(x)
    x = tf.keras.activations.relu(x)
    x = tf.math.l2_normalize(x, axis=-1)
    model = tf.keras.Model(inp, x, name="encoder")

    return model

def pathcer():
    inp = tf.keras.layers.Input(width)
    x = inp
    x = tf.keras.layers.Reshape((-1, width))(x)
    model = tf.keras.Model(inp,x,name="patcher")

    return model

"""
## The NNCLR model for contrastive pre-training
"""

class NNCLR(keras.Model):
    def __init__(
        self, temperature, queue_size,
    ):
        super(NNCLR, self).__init__()
        self.probe_accuracy = keras.metrics.CategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        self.contrastive_augmenter = augmenter(**contrastive_augmenter)
        self.classification_augmenter = augmenter(**classification_augmenter)
        self.encoder = encoder()
        self.patcher = pathcer()
        self.projection_head = keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width,
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(width,
                )
            ],
            name="projection_head",
        )
        self.linear_probe = keras.Sequential(
            [
            layers.Input(shape=(width,)),
            layers.Dense(width,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(width,
            )
            ], name="linear_probe"
        )
        self.temperature = temperature

        feature_dimensions = width
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super(NNCLR, self).compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0) [:queue_size]
        )
        return loss

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = tf.reshape(self.encoder(augmented_images_1),[-1, width])
            features_2 = tf.reshape(self.encoder(augmented_images_2), [-1, width])
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self.classification_augmenter(labeled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            features = tf.reshape(features, [-1,width])
            class_logits = self.linear_probe(features)

            labels = tf.reshape(labels, [-1,])
            class_logits, labels = filter_negatives(class_logits, labels)
            labels = tf.one_hot(tf.cast(labels, tf.int32), width)

            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}