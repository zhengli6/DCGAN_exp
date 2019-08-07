import os
import keras
import sys
from keras import layers
import keras.backend as K
import numpy as np 
import cv2
def deprocess_image(img):
    img = (img.astype(np.float64)-img.min())/ (img.max()-img.min()) # normalize the data to 0 - 1
    img = 255 * img # Now scale by 255
    return img.astype(np.uint8)
# Get a "l2 norm of gradients" tensor
def get_gradient_norm(model, weights):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

note = sys.argv[1]
latent_dim = 100
height = 32
width = 32
channels = 3

# Define possible opitmizer
adam_optimizer = keras.optimizers.Adam(0.0002, 0.5)
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0006, clipvalue=1.0, decay=1e-8)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0006, clipvalue=1.0, decay=1e-8)

######################### MODEL BEGIN #####################################

generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 8x8 128-channels feature map
x = layers.Dense(128 * 8 * 8, name='g_top_layer')(generator_input)
x = layers.BatchNormalization(momentum=0.5)(x)
x = layers.Activation("relu")(x)
x = layers.Reshape((8, 8, 128))(x)
# Upsampling
x = layers.UpSampling2D()(x)
x = layers.Conv2D(128, 5, padding='same', use_bias=True)(x)
x = layers.BatchNormalization(momentum=0.5)(x)
x = layers.Activation("relu")(x)
x = layers.UpSampling2D()(x)

x = layers.Conv2D(64, 5,  padding='same', use_bias=True)(x)
x = layers.BatchNormalization(momentum=0.5)(x)
x = layers.Activation("relu")(x)
# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 5, activation='tanh', padding='same', name='g_btm_layer')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

#################### Discriminator ##############################
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(32, 4, strides=2, name = 'd_top_layer')(discriminator_input)
x = layers.LeakyReLU(alpha=0.2)(x)
# x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, 4, strides=2, padding='same', use_bias=True)(x)
# x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
# x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=True)(x)
# x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU()(x)
# x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=True)(x)
# x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU()(x)

x = layers.Dropout(0.4)(x)
x1 = layers.Flatten()(x)
x = layers.Dense(1, activation='sigmoid', name = 'd_btm_layer')(x1)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Add gradients as anther outputs from training
discriminator.metrics_names.append("gradient_top_layer")
discriminator.metrics_tensors.append(get_gradient_norm(discriminator, discriminator.get_layer('d_top_layer').weights)) # top layer of generator

discriminator.metrics_names.append("gradient_bottom_layer")
discriminator.metrics_tensors.append(get_gradient_norm(discriminator, discriminator.get_layer('d_btm_layer').weights)) # bottom layer of generator


# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Add gradients as anther outputs from training
gan.metrics_names.append("gradient_top_layer")
gan.metrics_tensors.append(get_gradient_norm(gan, gan.layers[1].get_layer('g_top_layer').weights)) # top layer of generator

gan.metrics_names.append("gradient_bottom_layer")
gan.metrics_tensors.append(get_gradient_norm(gan, gan.layers[1].get_layer('g_btm_layer').weights)) # bottom layer of generator

######################### MODEL END #####################################

# =====================================Training==============================================
# Load CIFAR10 data
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
# Select images (class 7)
x_train = x_train[y_train.flatten() == 7]
# Normalize data
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 127.5 -1.
iterations = 5000
# mini batch size
batch_size = 128
save_dir = 'gan_images'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

training_img = np.ones((height*5+(10+1)*2,width*10+(10+1)*2,3))
generate_img = np.ones((height*5+(10+1)*2,width*10+(10+1)*2,3))
n = 0

print('%s\t%s\t%s\t%s\t%s' %('MODEL','STEP','LOSS','TOP_GRAD','BOTTOM_GRAD'))
# Start training loop
start = 0
generator_loss = []
discriminator_loss =[]
g_top_gradients =[]
g_btm_gradients =[]
d_top_gradients =[]
d_btm_gradients =[]
for step in range(iterations):
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # One side labeling smoothing
    # Add random noise to the labels - important trick!
    pos_labels = np.ones((batch_size, 1))
    pos_labels -= 0.1 * np.random.random(pos_labels.shape)

    neg_labels = np.zeros((batch_size, 1))
    neg_labels += 0.1 * np.random.random(pos_labels.shape)
    # flip labels 
    noise_prop = 0.05 # Randomly flip 5% of labels
    
    flipped_idx = np.random.choice(np.arange(len(pos_labels)), size=int(noise_prop*len(pos_labels)))
    pos_labels[flipped_idx] = 1 - pos_labels[flipped_idx]

    flipped_idx = np.random.choice(np.arange(len(neg_labels)), size=int(noise_prop*len(neg_labels)))
    neg_labels[flipped_idx] = 1 - neg_labels[flipped_idx]
    # Assemble labels discriminating real from fake images
    labels = np.concatenate([pos_labels,
                             neg_labels])

    # Train the discriminator
    discriminator.trainable = True
    d_loss, d_top_grad, d_btm_grad = discriminator.train_on_batch(combined_images, labels)
    # d_loss, d_top_grad, d_btm_grad = discriminator.train_on_batch(real_images, neg_labels)
    
    # d_loss, d_top_grad, d_btm_grad = discriminator.train_on_batch(generated_images, pos_labels)
    discriminator.trainable = False    

    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss, a_top_grad, a_btm_grad = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    # compute the gradients
    g_top_gradients.append(a_top_grad)
    g_btm_gradients.append(a_btm_grad)
    d_top_gradients.append(d_top_grad)
    d_btm_gradients.append(d_btm_grad)
    generator_loss.append(a_loss)
    discriminator_loss.append(d_loss)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    # Occasionally save / plot
    if step % 100 == 0:

        # Print metrics
        print('%s\t%s\t%.4f\t%.4f\t\t%.4f' %('DISC',step, d_loss,d_top_grad,d_btm_grad))
        print('%s\t%s\t%.4f\t%.4f\t\t%.4f' %('GENE',step, a_loss,a_top_grad,a_btm_grad))
        metrics = np.array([  generator_loss,
                              g_top_gradients,
                              g_btm_gradients,
                              discriminator_loss,
                              d_top_gradients,
                              d_btm_gradients
                             ])
        # save metrics for future analysis
        np.save('train_metrics_%s.npy'%note, metrics)
    if step % 100 == 0:
        # Save generated images
        i,j = np.unravel_index(n, (5,10))
        training_img[(i+1)*2+ i*height : (i+1)*2+ height*(i+1), (j+1)*2 + width*j : (j+1)*2 + width*(j+1) ]= generated_images[0]
        n+=1
        np.save('train_images_%s.npy'%note, training_img)

# Sample random points in the latent space
random_latent_vectors = np.random.normal(size=(50, latent_dim))
generated_images = generator.predict(random_latent_vectors)
for n in range(generated_images.shape[0]):
	i,j = np.unravel_index(n, (5,10))
	generate_img[(i+1)*2+ i*height : (i+1)*2+ height*(i+1), (j+1)*2 + width*j : (j+1)*2 + width*(j+1) ] = generated_images[n]

cv2.imwrite(os.path.join(save_dir, 'Training_%s.png'%note), deprocess_image(training_img))
cv2.imwrite(os.path.join(save_dir, 'Generated_%s.png'%note), deprocess_image(generate_img))

np.save('generate_images_%s.npy'%note, generated_images)