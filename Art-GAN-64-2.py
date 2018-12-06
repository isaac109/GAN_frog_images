#! /opt/rh/rh-python36/root/usr/bin/python
import keras
from keras import layers
import numpy as np

latent_dim = 64
height = 64
width = 64
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 16x16 128-channels feature map
x = layers.Dense(256*32*32)(generator_input)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Reshape((32,32,256))(x)

#Then, add a convolution layer
x = layers.Conv2D(512, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)

# Upsample to 32x32
x = layers.Conv2DTranspose(512, 4, strides = 2, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)

# Few more conv layers
x = layers.Conv2D(512, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(512, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)

#Produce a 32x32 1-channel reafture map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

#Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

#To Stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics = ['accuracy'])

#Set discriminator weights to non-trainable
# (will only apply to the 'gan' model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.summary()

import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

size = (64,64)
image_dir = '/home/isaac109@chapman.edu/resized_train_64'
images = []
for each in os.listdir(image_dir):
	images.append(os.path.join(image_dir,each))
print(len(images))
x_train = []
for each in images:
	img = image.load_img(each,size)
	try:
		x = image.img_to_array(img)
	except:
		continue
	x = preprocess_input(x)
	x_train.append(x)
x_train = np.asarray(x_train)
x_train = x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255.

iterations = 1001
batch_size = 20
save_dir = '/home/isaac109@chapman.edu/output-photos'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#start training loop
start = 0
for step in range(iterations):
	#Sample random points in the latent space
	random_latent_vectors = np.random.normal(0,1,size=(batch_size, latent_dim))
	
	# Decode them to fake images
	generated_images = generator.predict(random_latent_vectors)

	# Combine them with real images
	stop = start + batch_size
	real_images = x_train[start:stop]
	combined_images = np.concatenate([generated_images, real_images])

	#Assemble labels discriminating real from fake images
	labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
	#add random noise to the labels
	labels += 0.05 * np.random.random(labels.shape)

	#Train the discriminator
	d_loss = discriminator.train_on_batch(combined_images, labels)

	# sample random points in the latent space
	random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

	# Assemble labels that say "all real images"
	misleading_targets = np.zeros((batch_size, 1))

	#Train the generator (via the gan model,
	# where the discriminator weights are frozen)
	a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

	start += batch_size
	if start > len(x_train) - batch_size:
		start = 0
	if step % 100 == 0:
		gan.save_weights('/home/isaac109@chapman.edu/gan.h5')
		print('discriminator loss at step %s: %s' % (step, d_loss))
		print('adversarial loss at step %s: %s' % (step, a_loss))

		#Save one generated image
		picnum = 5
		for pic in range(picnum):
			img = image.array_to_img(generated_images[pic]*255.,scale=False)
			img.save(os.path.join(save_dir, 'generated_art' + str(step) + '-' + str(pic) + '.jpg'))
			img = image.array_to_img(real_images[pic] * 255., scale=False)
			img.save(os.path.join(save_dir, 'real_art' + str(step) + '-'+ str(pic) + '.jpg'))
