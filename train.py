from model import (
    CycleGan,
    make_discriminator,
    make_generator,
    generator_loss_fn,
    discriminator_loss_fn,
    GANMonitor,
)

import argparse
import mlflow
import tensorflow as tf

import tensorflow_datasets as tfds

# if needed
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


tfds.disable_progress_bar()

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 3)


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img


def main():
    # Todo change face dataset.  horse2zebra are for model test.
    autotune = tf.data.experimental.AUTOTUNE
    dataset, _ = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True)
    train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
    test_horses, test_zebras = dataset["testA"], dataset["testB"]

    buffer_size = 256
    batch_size = 1
    # Apply the preprocessing operations to the training data
    train_horses = (
        train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    train_zebras = (
        train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    # Apply the preprocessing operations to the test data
    test_horses = (
        test_horses.map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    test_zebras = (
        test_zebras.map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    disc_A = make_discriminator(input_img_size)
    disc_B = make_discriminator(input_img_size)
    gen_A2B = make_generator(input_img_size)
    gen_B2A = make_generator(input_img_size)

    cycle_gan_model = CycleGan(
        generator_A2B=gen_A2B,
        generator_B2A=gen_B2A,
        discriminator_A=disc_A,
        discriminator_B=disc_B,
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_A2B_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_B2A_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_A_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_B_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    # Callbacks
    plotter = GANMonitor()
    checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath
    )

    # Here we will train the model for just one epoch as each epoch takes around
    # 7 minutes on a single P100 backed machine.
    cycle_gan_model.fit(
        tf.data.Dataset.zip((train_horses, train_zebras)),
        epochs=1,
        callbacks=[plotter, model_checkpoint_callback],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mlflow", action="store_true", help="use mlflow")

    args = parser.parse_args()
    if args.mlflow:
        mlflow.tensorflow.autolog()
        with mlflow.start_run():
            main()
    else:
        main()
