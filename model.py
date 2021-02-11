from tensorflow.keras.layers import (
    Conv2D,
    Input,
    LeakyReLU,
    add,
    Conv2DTranspose,
    Activation,
)

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def encodeblock(inputs):

    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    return x


def residualblock(inputs):
    """Some Information about ResidualBlock"""

    x = ReflectionPadding2D()(inputs)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tfa.layers.InstanceNormalization()(x)
    outputs = add([inputs, x])
    return outputs


def decodeblock(inputs):

    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same")(inputs)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = Activation("relu")(x)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = Conv2D(3, kernel_size=(7, 7), strides=(1, 1))(x)
    outputs = Activation("tanh")(x)
    return outputs


def make_generator(img_shape):
    num_res_block = 9

    inputs = Input(shape=img_shape)

    x = encodeblock(inputs)

    for _ in range(num_res_block):
        x = residualblock(x)

    x = decodeblock(x)

    generator = tf.keras.models.Model(inputs=inputs, outputs=x)

    return generator


def make_discriminator(img_shape):

    inputs = Input(shape=img_shape)

    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(1, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)

    discriminator = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return discriminator


class CycleGan(tf.keras.Model):
    def __init__(
        self,
        generator_A2B,
        generator_B2A,
        discriminator_A,
        discriminator_B,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_A2B = generator_A2B
        self.gen_B2A = generator_B2A
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_A2B_optimizer,
        gen_B2A_optimizer,
        disc_A_optimizer,
        disc_B_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_A2B_optimizer = gen_A2B_optimizer
        self.gen_B2A_optimizer = gen_B2A_optimizer
        self.disc_A_optimizer = disc_A_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_A, real_B = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_B = self.gen_A2B(real_A, training=True)
            # Zebra to fake horse -> y2x
            fake_A = self.gen_B2A(real_B, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_A = self.gen_B2A(fake_B, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_B = self.gen_A2B(fake_A, training=True)

            # Identity mapping
            same_A = self.gen_B2A(real_A, training=True)
            same_B = self.gen_A2B(real_B, training=True)

            # Discriminator output
            disc_real_A = self.disc_A(real_A, training=True)
            disc_fake_A = self.disc_A(fake_A, training=True)

            disc_real_B = self.disc_B(real_B, training=True)
            disc_fake_B = self.disc_B(fake_B, training=True)

            # Generator adverserial loss
            gen_A2B_loss = self.generator_loss_fn(disc_fake_B)
            gen_B2A_loss = self.generator_loss_fn(disc_fake_A)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_B, cycled_B) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_A, cycled_A) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_B, same_B)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_A, same_A)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_A2B_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_B2A_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_A_loss = self.discriminator_loss_fn(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss_fn(disc_real_B, disc_fake_B)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_A2B.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_B2A.trainable_variables)

        # Get the gradients for the discriminators
        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        # Update the weights of the generators
        self.gen_A2B_optimizer.apply_gradients(
            zip(grads_G, self.gen_A2B.trainable_variables)
        )
        self.gen_B2A_optimizer.apply_gradients(
            zip(grads_F, self.gen_B2A.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_A_optimizer.apply_gradients(
            zip(disc_A_grads, self.disc_A.trainable_variables)
        )
        self.disc_B_optimizer.apply_gradients(
            zip(disc_B_grads, self.disc_B.trainable_variables)
        )

        return {
            "A2B_loss": total_loss_G,
            "B2A_loss": total_loss_F,
            "D_A_loss": disc_A_loss,
            "D_B_loss": disc_B_loss,
        }


# Todo change face dataset
class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_horses.take(self.num_img)):
            prediction = self.model.gen_A2B(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()


# Loss function for evaluating adversarial loss
adv_loss_fn = tf.keras.losses.MeanSquaredError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5
