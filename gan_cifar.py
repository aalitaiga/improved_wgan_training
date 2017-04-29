import os
import time
from datetime import date

import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

path = '/Tmp/alitaiga/ift6266/wgan_{}'.format(date.today())
if not os.path.exists(path):
    os.makedirs(path)

perso = '/Users/Adrien/Repositories/IFT6266h17/'
server = 'Tmp/alitaiga/ift6266/'
path = path

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 32 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 32 # Batch size
ITERS = 500000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, z, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 32])

    output = lib.ops.conv2d.Conv2D('Generator.1', 3, DIM, 4, z, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN11', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Generator.11', DIM, DIM, 4, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN12', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Generator.11', DIM, DIM, 3, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN13', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Generator.11', DIM, 4*DIM, 3, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN14', [0,2,3], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [BATCH_SIZE, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return output

def Discriminator(output):
    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.15', DIM, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [BATCH_SIZE, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [BATCH_SIZE])

real_data_center = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 32, 32])
real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
real_data_int = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
real_data_center = 2*((tf.cast(real_data_center, tf.float32)/255.)-.5)

fake_center = Generator(BATCH_SIZE, z=real_data_int)

padding = [[0, 0], [0, 0], [16, 16], [16, 16]]
real_data = real_data_int + tf.pad(real_data_center, padding, "CONSTANT") 
disc_real = Discriminator(real_data)

fake_data = real_data_int + tf.pad(fake_center, padding, "CONSTANT") 
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')


# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1, 1, 1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# For generating samples
#fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
#fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(itera, ext):
    samples = session.run([fake_data], feed_dict={real_data_int: ext})
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'u/alitaiga/repositories/samples/'+'mscoc_samples_{}.jpg'.format(itera))

# Dataset iterators
coco_train = H5PYDataset(path + 'coco_cropped.h5', which_sets=('train',))
coco_test = H5PYDataset(path + 'coco_cropped.h5', which_sets=('valid',))

train_stream = DataStream(
    coco_train,
    iteration_scheme=ShuffledScheme(coco_train.num_examples, BATCH_SIZE)
)

test_stream = DataStream(
    coco_test,
    iteration_scheme=ShuffledScheme(coco_test.num_examples, BATCH_SIZE)
)
test_iter = test_stream.get_epoch_iterator()
_data = next(test_iter)

saver = tf.train.Saver()

# Train loop
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = train_stream.get_epoch_iterator()
    print "Starting training"

    for iteration in xrange(500000):
        start_time = time.time()
        # Train generator
        try:
            for i in xrange(CRITIC_ITERS):
                _ext, _center = next(gen)
                if _ext.shape[0] != BATCH_SIZE:
                    raise StopIteration
                _disc_cost, _ = session.run([disc_cost, disc_train_op],
                    feed_dict={real_data_int: _ext, real_data_center: _center})
        except StopIteration:
            gen = train_stream.get_epoch_iterator()
            continue

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={real_data_int: _ext, real_data_center: _center})

        print 'iteration: {},  train disc cost: {}, time: {}'.format(iteration, _disc_cost, time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 10000 == 0:
            generate_image(iteration, _data)

        if iteration % 5000:
            saver.save(session, path + '/params_' + 'ift6266_gan.ckpt')
