import tensorflow as tf
from matplotlib import pyplot as plt


class Sigma():

    @staticmethod
    def load_and_preprocess_single_image(image_path, img_height=512, img_width=512):
        # Read and decode the image from the file path
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize the image to the required dimensions
        img = tf.image.resize(img, [img_height, img_width])
        
        # Normalize the image to the [-1, 1] range
        img = (img / 127.5) - 1
        
        # Ensure the image has the correct data type
        img = tf.cast(img, tf.float32)
        
        return img
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    @staticmethod
    def generator_loss(disc_generated_output, gen_output, target):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (100 * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def generator(self):
        inputs = tf.keras.layers.Input(shape=[512, 512, 3])
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4)
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
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def generate_single_image(self, generator, img_path, img_name):
    
        # Load and preprocess the image
        input_image = self.load_and_preprocess_single_image(img_path)

        # Reshape the input image to include batch dimension
        input_image = tf.expand_dims(input_image, axis=0)

        # Use the generator to make a prediction
        prediction = generator(input_image, training=True)

        # # Visualize the input image, ground truth, and predicted image
        # plt.figure(figsize=(10, 10))
        # display_list = [input_image[0], prediction[0]] 
        # title = ['Input Image', 'Predicted Image']
        # for i in range(2):
        #     plt.subplot(1, 2, i + 1)
        #     plt.title(title[i])
        #     plt.imshow((display_list[i] * 0.5 + 0.5).numpy())
        #     plt.axis('off')
        # plt.show()

        # Plot the predicted image only
        plt.figure(figsize=(8, 8))
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow((prediction[0] * 0.5 + 0.5).numpy())
        plt.axis('off')
        
        # Save the plot as an image
        output_path = f"test_images/result_{img_name}.jpg"
        plt.savefig(output_path)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


sigma = Sigma()
generator = sigma.generator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
checkpoint_dir = './training_checkpoints'
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

img_path = 'C:\\Users\\DEBIEMASL\\OneDrive - Durr Group\\Dokumente\\Projekte\\Studienarbeit\\test_images\\drawing2.jpeg'
img_name = "drawing2"
sigma.generate_single_image(generator, img_path, img_name)