import tensorflow as tf
import os
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

class Sigma:
  def __init__(self, dataset_path = "dataset cars/"):
    self.dataset_path = dataset_path

    # The facade training set consist of 400 images
    self.BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    self.BATCH_SIZE = 1
    # Each image is 512x512 in size
    self.IMG_WIDTH = 512
    self.IMG_HEIGHT = 512
    
    # output channels of images in dataset ----
    self.OUTPUT_CHANNELS = 3

    # ------------------------------------------
    self.LAMBDA = 100
    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # define Tensorflow generator- and discriminator-optimizer
    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Checkpoint stuff
    self.checkpoint_dir = './training_checkpoints'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator(),
                                    discriminator=self.discriminator())
    
    # Logs -------------------------------
    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  @staticmethod
  def load(image_file): # image file is the drawn image
      # Read and decode an image file to a uint8 tensor
      input_image = tf.io.read_file(image_file)
      input_image = tf.io.decode_jpeg(input_image)

      # Convert the image file to a string
      image_file_str = tf.strings.as_string(image_file)  # Convert the symbolic tensor to a string
      
      # Replace "drawn" with "original" in the filename string
      real_image_str = tf.strings.regex_replace(image_file_str, "drawn", "original")
      
      # Convert the string filename back to a symbolic tensor
      real_image = tf.io.read_file(real_image_str)
      real_image = tf.io.decode_jpeg(real_image)

      # Ensure that both images have the same number of channels
      # For example, if one image has 4 channels, convert it to 3 channels
      input_image = input_image[..., :3]  # Keep only the first 3 channels if there are 4
      real_image = real_image[..., :3]  # Keep only the first 3 channels if there are 4
      
      # Convert both images to float32 tensors
      input_image = tf.cast(input_image, tf.float32)
      real_image = tf.cast(real_image, tf.float32)

      return input_image, real_image

  @staticmethod
  def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

  def random_crop(self, input_image, real_image): 
    stacked_image = tf.stack([input_image, real_image], axis=0)
    
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

  # Normalizing the images to [-1, 1]
  @staticmethod
  def normalize(input_image, real_image):
    input_image = (input_image / 256) - 1
    real_image = (real_image / 256) - 1

    return input_image, real_image

  @tf.function()
  def random_jitter(self, input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = self.resize(input_image, real_image, 572, 572)

    # Random cropping back to 256x256
    input_image, real_image = self.random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
      # Random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


  def load_image_train(self, image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.random_jitter(input_image, real_image)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image

  def load_image_test(self, image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.resize(input_image, real_image,
                                    self.IMG_HEIGHT, self.IMG_WIDTH)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image

  # ## Build input pipeline with tf.data
  def get_train_dataset(self):
      # Get a list of JPG files that contain drawings
      drawing_files = [os.path.join('dataset/train/', file) for file in os.listdir('dataset/train/') if file.endswith('.jpg') and 'drawn' in file]

      # Create a dataset from the filtered list of files
      train_dataset = tf.data.Dataset.from_tensor_slices(drawing_files)

      train_dataset = train_dataset.map(self.load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
      train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
      train_dataset = train_dataset.batch(self.BATCH_SIZE)
      return train_dataset

  def get_test_dataset(self):
      # Get a list of JPG files that contain drawings
      drawing_files = [os.path.join('dataset/test/', file) for file in os.listdir('dataset/test/') if file.endswith('.jpg') and 'drawn' in file]
      
      # Create a dataset from the filtered list of files
      test_dataset = tf.data.Dataset.from_tensor_slices(drawing_files)

      test_dataset = test_dataset.map(self.load_image_test)
      test_dataset = test_dataset.batch(self.BATCH_SIZE)

      return test_dataset

  @staticmethod
  def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


  @staticmethod
  def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


  def generator(self, training=True):
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
      self.downsample(64, 4, apply_batchnorm=False),    # (batch_size, 256, 256, 64)
      self.downsample(128, 4),                           # (batch_size, 128, 128, 128)
      self.downsample(256, 4),                           # (batch_size, 64, 64, 256)
      self.downsample(512, 4),                           # (batch_size, 32, 32, 512)
      self.downsample(512, 4),                           # (batch_size, 16, 16, 512)
      self.downsample(512, 4),                           # (batch_size, 8, 8, 512)
      self.downsample(512, 4),                           # (batch_size, 4, 4, 512)
      self.downsample(512, 4),                           # (batch_size, 2, 2, 512)
      self.downsample(512, 4),                           # (batch_size, 1, 1, 512)
    ]

    up_stack = [
      self.upsample(512, 4, apply_dropout=True),        # (batch_size, 2, 2, 1024)
      self.upsample(512, 4, apply_dropout=True),        # (batch_size, 4, 4, 1024)
      self.upsample(512, 4, apply_dropout=True),        # (batch_size, 8, 8, 1024)
      self.upsample(512, 4, apply_dropout=True),        # (batch_size, 8, 8, 1024)
      self.upsample(512, 4),                            # (batch_size, 16, 16, 1024)
      self.upsample(256, 4),                            # (batch_size, 32, 32, 512)
      self.upsample(128, 4),                            # (batch_size, 64, 64, 256)
      self.upsample(64, 4),                             # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

  def generator_loss(self, disc_generated_output, gen_output, target):
    gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

  def discriminator(self):
      initializer = tf.random_normal_initializer(0., 0.02)

      inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
      tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

      x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 512, 512, channels*2)

      down1 = self.downsample(64, 4, False)(x)  # (batch_size, 256, 256, 64)
      down2 = self.downsample(128, 4)(down1)  # (batch_size, 128, 128, 128)
      down3 = self.downsample(256, 4)(down2)  # (batch_size, 64, 64, 256)
      down4 = self.downsample(512, 4)(down3)  # (batch_size, 64, 64, 256)

      zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 66, 66, 256)
      conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 63, 63, 512)

      batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

      leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

      zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 65, 65, 512)

      last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 62, 62, 1)

      return tf.keras.Model(inputs=[inp, tar], outputs=last)

  def discriminator_loss(self, disc_real_output, disc_generated_output):
    real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  @staticmethod
  def generate_images(model, test_input, tar, imgid=0):
      prediction = model(test_input)
      
      # convert to list if it is a single value
      if isinstance(prediction, list):
         print(f"Prediction lenght: {len(prediction)}")
      else:
         prediction = [prediction] 

      plt.figure(figsize=(15, 15))
      if tar is not None:
          display_list = [test_input[0], tar[0], prediction[0]]
          title = ['Input Image', 'Ground Truth', 'Predicted Image']
      else:
          display_list = [test_input[0], prediction[0]]
          title = ['Input Image', 'Predicted Image']

      for i in range(len(display_list)):
          plt.subplot(1, len(display_list), i + 1)
          plt.title(title[i])
          plt.imshow(display_list[i] * 0.5 + 0.5)
          plt.axis('off')

      output_path = f"results/result_{imgid}.jpg"
      plt.savefig(output_path)
      plt.show()

      return prediction

  @tf.function
  def train_step(self, input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self.generator(input_image, training=True)

      disc_real_output = self.discriminator([input_image, target], training=True)
      disc_generated_output = self.discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
      disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                            self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))

    with self.summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

  def fit(self, train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step) % 1000 == 0:
        display.clear_output(wait=True)

        if step != 0:
          print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

        start = time.time()

        self.generate_images(self.generator, example_input, example_target, step)
        print(f"Step: {step//1000}k")

      self.train_step(input_image, target, step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)


      # Save (checkpoint) the model every 5k steps
      if (step + 1) % 5000 == 0:
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)


if __name__ == "__main__":
    sigma_obj = Sigma()
    test_dataset = sigma_obj.get_test_dataset()

    # Only for training---------------------------------------------------
    train_dataset = sigma_obj.get_train_dataset()
    sigma_obj.fit(train_dataset, test_dataset, steps=40000)
    # Without Training ---------------------------------------------------
    # Restoring the latest checkpoint in checkpoint_dir
    sigma_obj.checkpoint.restore(tf.train.latest_checkpoint(sigma_obj.checkpoint_dir))
    # --------------------------------------------------------------------

    # Generate some images
    loopcounter = 5
    for inp, tar in test_dataset.take(5):
        sigma_obj.generate_images(sigma_obj.generator, inp, tar, f"test{loopcounter}")
        loopcounter += 1

    # Generate single image (for Website)
    sigma_obj.generate_images(sigma_obj.generator, inp, tar, f"test{loopcounter}")

