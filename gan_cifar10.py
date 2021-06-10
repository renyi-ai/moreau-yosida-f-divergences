import argparse
from functools import partial
import math
import os
import numpy as np
import pickle
import random
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.ops import array_ops


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# DATA
class ImageDataSet(object):
    def __init__(self, images):
        assert images.ndim == 4

        self.num_examples = images.shape[0]

        self.images = images
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end]


def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


def get_cifar10_dataset(split=None):
    train_dir = "data/cifar10/"

    data = []
    for i in range(1, 7):
        if i < 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        elif i == 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'test_batch')
        dct = unpickle(path)
        data.append(dct[b'data'])

    data_arr = np.concatenate(data, axis=0)
    raw_float = np.array(data_arr, dtype='float32') / 256.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])

    if split is None:
        pass
    elif split == 'train':
        images = images[:-10000]
    elif split == 'test':
        images = images[-10000:]
    else:
        raise ValueError('unknown split')

    dataset = ImageDataSet(images)

    return dataset


def random_flip(x, up_down=False, left_right=True):
    with tf.name_scope('random_flip'):
        s = tf.shape(x)
        if up_down:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[1]))
        if left_right:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        return x


def get_cifar10_tf(batch_size=1, shape=[32, 32], split=None, augment=True, start_queue_runner=True):
    with tf.name_scope('get_cifar10_tf'):
        dataset = get_cifar10_dataset(split=split)

        images = tf.constant(dataset.images, dtype='float32')

        image = tf.train.slice_input_producer([images], shuffle=True)

        images_batch = tf.train.batch(image, batch_size=batch_size, num_threads=8)

        if augment:
            images_batch = random_flip(images_batch)
            images_batch += tf.random_uniform(tf.shape(images_batch), 0.0, 1.0/256.0)

        if shape != [32, 32]:
            images_batch = tf.image.resize_bilinear(images_batch, [shape[0], shape[1]])

        if start_queue_runner:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

        return images_batch


def image_grid(x, size=8):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]


def image_grid_summary(name, x):
    with tf.name_scope(name):
        tf.summary.image('grid', image_grid(x))


def scalars_summary(name, x):
    with tf.name_scope(name):
        x = tf.reshape(x, [-1])
        mean, var = tf.nn.moments(x, axes=0)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', tf.sqrt(var))
        tf.summary.scalar('min', tf.reduce_min(x))
        tf.summary.scalar('max', tf.reduce_max(x))


class EMAHelper(object):
    def __init__(self, decay=0.99, session=None):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)
        self.apply = self.ema.apply(self.all_vars)
        self.averages = [self.ema.average(var) for var in self.all_vars]

    def average_dict(self):
        ema_averages_results = self.session.run(self.averages)
        return {var: value for var, value in
                zip(self.all_vars, ema_averages_results)}

    def variables_to_restore(self):
        return self.ema.variables_to_restore(tf.moving_average_variables())


def apply_conv(x, filters=32, kernel_size=3, he_init=True):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(
        x, filters=filters, kernel_size=kernel_size, padding='SAME', kernel_initializer=initializer
    )


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(
        x,
        decay=0.9,
        center=True,
        scale=True,
        epsilon=1e-5,
        zero_debias_moving_mean=True,
        is_training=is_training
    )


def stable_norm(x):
    return tf.norm(tf.contrib.layers.flatten(x), ord=2, axis=1, keepdims=True)


def normalize(x):
    return x / tf.maximum(tf.expand_dims(tf.expand_dims(stable_norm(x), -1), -1), 1e-10)


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n(
            [x[:,::2,::2,:], x[:,1::2,::2,:], x[:,::2,1::2,:], x[:,1::2,1::2,:]]
        ) / 4


def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))


def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)


def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)


def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = partial(apply_conv, filters=filters)
        conv_2 = partial(conv_meanpool, filters=filters)
        conv_shortcut = partial(conv_meanpool, filters=filters, kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = partial(upsample_conv, filters=filters)
        conv_2 = partial(apply_conv, filters=filters)
        conv_shortcut = partial(upsample_conv, filters=filters, kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = partial(apply_conv, filters=filters)
        conv_2 = partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)

        skip = meanpool_conv(x, filters=filters, kernel_size=1, he_init=False)
        return skip + update


def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        channels = 128
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * channels)
            x = tf.reshape(z, [-1, 4, 4, channels])

        with tf.name_scope('x1'):
            x = resblock(x, filters=channels, resample='up', normalize=True) # 8
            x = resblock(x, filters=channels, resample='up', normalize=True) # 16
            x = resblock(x, filters=channels, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='down') # 8
            x = resblock(x, filters=128) # 16
            x = resblock(x, filters=128) # 32

        with tf.name_scope('post_process'):
            x = activation(x)
            x = tf.reduce_mean(x, axis=[1, 2])
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)
            return flat


def quotient_discriminator(x, reuse):
    validity = discriminator(x, reuse)
    return validity - discriminator(tf.zeros_like(x[0:1]), True)


def conjugate_trivial(f_x):
    return tf.reduce_mean(f_x)


@tf.custom_gradient
def conjugate_kl(f_x):
    max_f_x = tf.reduce_max(f_x)
    def grad_conjugate_kl(dy):
        return dy * (tf.exp(f_x - max_f_x) / tf.reduce_sum(tf.exp(f_x - max_f_x)))
    return (
        tf.log(tf.reduce_mean(tf.exp(f_x - max_f_x))) + max_f_x,
        grad_conjugate_kl
    )


@tf.custom_gradient
def gamma_reverse_kl(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(1 / (1 - f_x + gamma)) + 1
        F_gamma_derivative = tf.reduce_mean(1 / ((1 - f_x + gamma) ** 2))
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_max(f_x) - 1 + 0.01
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_reverse_kl(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (1 / ((1 - f_x + gamma) ** 2))
        F_gamma_derivative = tf.reduce_mean(1 / ((1 - f_x + gamma) ** 2))
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_reverse_kl
    )


def conjugate_reverse_kl(f_x):
    gamma = gamma_reverse_kl(f_x)
    return tf.reduce_mean(-tf.log(1 - f_x + gamma)) + gamma


@tf.custom_gradient
def gamma_chi2(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(
            ((f_x - gamma) / 2 + 1) * tf.cast(f_x - gamma >= -2, tf.float32)
        ) + 1
        F_gamma_derivative = tf.reduce_mean(
            tf.cast(f_x - gamma >= -2, tf.float32) / 2
        )
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_mean(f_x)
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_chi2(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (
            tf.cast(f_x - gamma >= -2, tf.float32) / 2
        )
        F_gamma_derivative = tf.reduce_mean(
            tf.cast(f_x - gamma >= -2, tf.float32) / 2
        )
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_chi2
    )


def conjugate_chi2(f_x):
    gamma = gamma_chi2(f_x)
    return tf.reduce_mean(
        -1 * tf.cast(f_x - gamma < -2, tf.float32) +
        (((f_x - gamma) ** 2) / 4 + f_x - gamma) * tf.cast(f_x - gamma >= -2, tf.float32)
    ) + gamma


@tf.custom_gradient
def gamma_reverse_chi2(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(1 / tf.sqrt(1 - f_x + gamma)) + 1
        F_gamma_derivative = tf.reduce_mean(1 / (2 * (tf.sqrt(1 - f_x + gamma) ** 3)))
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_max(f_x) - 1 + 1e-4
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_reverse_chi2(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (1 / (2 * (tf.sqrt(1 - f_x + gamma) ** 3)))
        F_gamma_derivative = tf.reduce_mean(1 / (2 * (tf.sqrt(1 - f_x + gamma) ** 3)))
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_reverse_chi2
    )


def conjugate_reverse_chi2(f_x):
    gamma = gamma_reverse_chi2(f_x)
    return tf.reduce_mean(2 - 2 * tf.sqrt(1 - f_x + gamma)) + gamma


@tf.custom_gradient
def gamma_hellinger2(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(1 / ((1 - f_x + gamma) ** 2)) + 1
        F_gamma_derivative = tf.reduce_mean(2 / ((1 - f_x + gamma) ** 3))
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_max(f_x) - 1 + 0.1
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_hellinger2(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (2 / ((1 - f_x + gamma) ** 3))
        F_gamma_derivative = tf.reduce_mean(2 / ((1 - f_x + gamma) ** 3))
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_hellinger2
    )


def conjugate_hellinger2(f_x):
    gamma = gamma_hellinger2(f_x)
    return tf.reduce_mean((f_x - gamma) / (1 - f_x + gamma)) + gamma


@tf.custom_gradient
def gamma_js(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(1 / (2 * tf.exp(gamma - f_x) - 1)) + 1
        F_gamma_derivative = tf.reduce_mean((2 * tf.exp(f_x - gamma)) / ((tf.exp(f_x - gamma) - 2) ** 2))
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_max(f_x) - math.log(2) + 1e-5
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_js(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (
            (2 * tf.exp(f_x - gamma)) / ((tf.exp(f_x - gamma) - 2) ** 2)
        )
        F_gamma_derivative = tf.reduce_mean((2 * tf.exp(f_x - gamma)) / ((tf.exp(f_x - gamma) - 2) ** 2))
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_js
    )


def conjugate_js(f_x):
    gamma = gamma_js(f_x)
    return tf.reduce_mean(-tf.log(2 - tf.exp(f_x - gamma))) + gamma


@tf.custom_gradient
def lambertw(z):

    def cond(w, step):
        return tf.greater(tf.reduce_max(tf.abs(step)), 1e-6)

    def body(w, step):
        step = (w * tf.exp(w) - z) / (tf.exp(w) + w * tf.exp(w))
        w = w - step
        return w, step

    w = tf.log(1 + z)
    step = tf.ones_like(w)

    w, step = tf.while_loop(
        cond, body, (w, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_lambertw(dy):
        return dy * (w / (z * (1 + w)))

    return (
        w,
        grad_lambertw
    )


@tf.custom_gradient
def gamma_jeffreys(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        w = lambertw(tf.exp(1 - f_x + gamma))
        F_gamma = -tf.reduce_mean(1 / w) + 1
        F_gamma_derivative = tf.reduce_mean(
            (1 / w) - (1 / (w + 1))
        )
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_mean(f_x)
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_jeffreys(dy):
        w = lambertw(tf.exp(1 - f_x + gamma))
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (
            (1 / w) - (1 / (w + 1))
        )
        F_gamma_derivative = tf.reduce_mean(
            (1 / w) - (1 / (w + 1))
        )
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_jeffreys
    )


def conjugate_jeffreys(f_x):
    gamma = gamma_jeffreys(f_x)
    w = lambertw(tf.exp(1 - f_x + gamma))
    return tf.reduce_mean(f_x - gamma + w + 1 / w - 2) + gamma


@tf.custom_gradient
def gamma_triangular(f_x):

    def cond(gamma, step):
        return tf.greater(tf.abs(step), 1e-6)

    def body(gamma, step):
        F_gamma = -tf.reduce_mean(
            (2 / ((1 - f_x + gamma) ** (1/2)) - 1) * tf.cast(f_x - gamma >= -3, tf.float32)
        ) + 1
        F_gamma_derivative = tf.reduce_mean(
            (1 / (((1 - f_x + gamma) ** (1/2)) ** 3)) * tf.cast(f_x - gamma >= -3, tf.float32)
        )
        step = F_gamma / F_gamma_derivative
        gamma = gamma - step
        return gamma, step

    gamma = tf.reduce_max(f_x) - 1 + 1e-5
    step = tf.ones_like(gamma)

    gamma, step = tf.while_loop(
        cond, body, (gamma, step), back_prop=False,
        maximum_iterations=10000
    )

    def grad_gamma_triangular(dy):
        F_gamma_gradient = -(1 / tf.cast(tf.shape(f_x)[0], tf.float32)) * (
                1 / (((1 - f_x + gamma) ** (1/2)) ** 3)
        ) * tf.cast(f_x - gamma >= -3, tf.float32)
        F_gamma_derivative = tf.reduce_mean(
            (1 / (((1 - f_x + gamma) ** (1/2)) ** 3)) * tf.cast(f_x - gamma >= -3, tf.float32)
        )
        return dy * (-F_gamma_gradient / F_gamma_derivative)

    return (
        gamma,
        grad_gamma_triangular
    )


def conjugate_triangular(f_x):
    gamma = gamma_triangular(f_x)
    return tf.reduce_mean(
        -1 * tf.cast(f_x - gamma < -3, tf.float32) +
        ((1 - f_x + gamma) ** (1 / 2) - 1) * ((1 - f_x + gamma) ** (1 / 2) - 3) * tf.cast(f_x - gamma >= -3, tf.float32)
    ) + gamma


def conjugate_tv(f_x):
    gamma = tf.reduce_max(f_x) - 1
    return tf.reduce_mean(
        -1 * tf.cast(f_x - gamma < -1, tf.float32) +
        (f_x - gamma) * tf.cast(f_x - gamma >= -1, tf.float32)
    ) + gamma


def loose_conjugate_kl(f_x):
    return tf.reduce_mean(tf.exp(f_x) - 1)


def loose_discriminator_kl(x, reuse):
    validity = discriminator(x, reuse)
    return validity


def loose_conjugate_reverse_kl(f_x):
    return tf.reduce_mean(-tf.log(1 - f_x))


def loose_discriminator_reverse_kl(x, reuse):
    validity = discriminator(x, reuse)
    return 1 - tf.exp(-validity)


def loose_conjugate_chi2(f_x):
    return tf.reduce_mean((f_x ** 2) / 4 + f_x)


def loose_discriminator_chi2(x, reuse):
    validity = discriminator(x, reuse)
    return validity


def loose_conjugate_reverse_chi2(f_x):
    return tf.reduce_mean(2 - 2 * tf.sqrt(1 - f_x))


def loose_discriminator_reverse_chi2(x, reuse):
    validity = discriminator(x, reuse)
    return 1 - tf.exp(-validity)


def loose_conjugate_hellinger2(f_x):
    return tf.reduce_mean(f_x / (1 - f_x))


def loose_discriminator_hellinger2(x, reuse):
    validity = discriminator(x, reuse)
    return 1 - tf.exp(-validity)


def loose_conjugate_js(f_x):
    return tf.reduce_mean(-tf.log(2 - tf.exp(f_x)))


def loose_discriminator_js(x, reuse):
    validity = discriminator(x, reuse)
    return math.log(2) - tf.exp(-validity)


def loose_conjugate_jeffreys(f_x):
    w = lambertw(tf.exp(1 - f_x))
    return tf.reduce_mean(f_x + w + 1 / w - 2)


def loose_discriminator_jeffreys(x, reuse):
    validity = discriminator(x, reuse)
    return validity


def loose_conjugate_triangular(f_x):
    return tf.reduce_mean(
        -1 * tf.cast(f_x < -3, tf.float32) +
        (5 - 4 * ((1 - f_x) ** (1 / 2)) - f_x) * tf.cast(f_x >= -3, tf.float32)
    )


def loose_discriminator_triangular(x, reuse):
    validity = discriminator(x, reuse)
    return 1 - tf.exp(-validity)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                   default="logs/gan_cifar10/")
    parser.add_argument("--run_name",                  default="")
    parser.add_argument("--log_freq",      type=int,   default=100)
    parser.add_argument("--iterations",    type=int,   default=100000)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--val_freq",      type=int,   default=1000)
    parser.add_argument("--val_size",      type=int,   default=100)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--b1",            type=float, default=0.0)
    parser.add_argument("--b2",            type=float, default=0.9)
    parser.add_argument("--ema_decay",     type=float, default=0.9999)
    parser.add_argument("--lambda_gp",     type=float, default=10)
    parser.add_argument("--alpha",         type=float, default=float("inf"))
    parser.add_argument("--init_beta",     type=float, default=1)
    parser.add_argument("--final_beta",    type=float, default=1)
    parser.add_argument("--K",             type=float, default=1)
    parser.add_argument("--n_critic",      type=int,   default=5)
    parser.add_argument("--discriminator",             default="quotient", choices=[
        "base", "quotient"
    ])
    parser.add_argument("--divergence",                default="trivial", choices=[
        "trivial", "kl", "reverse_kl", "chi2", "reverse_chi2", "hellinger2", "js", "jeffreys", "triangular", "tv"
    ])
    parser.add_argument('--forward', dest='forward', action='store_true')
    parser.add_argument('--reverse', dest='forward', action='store_false')
    parser.set_defaults(forward=True)
    parser.add_argument('--penalized', dest='penalized', action='store_true')
    parser.add_argument('--non_penalized', dest='penalized', action='store_false')
    parser.set_defaults(penalized=True)
    parser.add_argument('--decay_lr', dest='decay_lr', action='store_true')
    parser.add_argument('--const_lr', dest='decay_lr', action='store_false')
    parser.set_defaults(decay_lr=True)
    parser.add_argument('--tight', dest='tight', action='store_true')
    parser.add_argument('--loose', dest='tight', action='store_false')
    parser.set_defaults(tight=True)
    args = parser.parse_args()
    print(args)

    # set seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    random.seed(args.random_seed)

    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=config)

    log_dir = args.log_dir + args.run_name
    os.makedirs(log_dir)

    if args.tight:
        discriminator_fn = {
            "base": discriminator,
            "quotient": quotient_discriminator,
        }[args.discriminator]

        conjugate_fn = lambda f_x: {
            "trivial": conjugate_trivial,
            "kl": conjugate_kl,
            "reverse_kl": conjugate_reverse_kl,
            "chi2": conjugate_chi2,
            "reverse_chi2": conjugate_reverse_chi2,
            "hellinger2": conjugate_hellinger2,
            "js": conjugate_js,
            "jeffreys": conjugate_jeffreys,
            "triangular": conjugate_triangular,
            "tv": conjugate_tv,
        }[args.divergence](f_x - tf.reduce_max(f_x)) + tf.reduce_max(f_x)
    else:
        discriminator_fn = {
            "kl": loose_discriminator_kl,
            "reverse_kl": loose_discriminator_reverse_kl,
            "chi2": loose_discriminator_chi2,
            "reverse_chi2": loose_discriminator_reverse_chi2,
            "hellinger2": loose_discriminator_hellinger2,
            "js": loose_discriminator_js,
            "jeffreys": loose_discriminator_jeffreys,
            "triangular": loose_discriminator_triangular,
        }[args.divergence]

        conjugate_fn = {
            "kl": loose_conjugate_kl,
            "reverse_kl": loose_conjugate_reverse_kl,
            "chi2": loose_conjugate_chi2,
            "reverse_chi2": loose_conjugate_reverse_chi2,
            "hellinger2": loose_conjugate_hellinger2,
            "js": loose_conjugate_js,
            "jeffreys": loose_conjugate_jeffreys,
            "triangular": loose_conjugate_triangular,
        }[args.divergence]


    global_step = tf.Variable(0, trainable=False, name='global_step')

    with tf.name_scope('placeholders'):
        x_true_ph = get_cifar10_tf(batch_size=args.batch_size)
        x_10k_ph = get_cifar10_tf(batch_size=10000)
        x_50k_ph = get_cifar10_tf(batch_size=50000)

        is_training = tf.placeholder(bool, name='is_training')
        use_agumentation = tf.identity(is_training, name='is_training')

    with tf.name_scope('pre_process'):
        x_true = (x_true_ph - 0.5) * 2.0

        x_10k = (x_10k_ph - 0.5) * 2.0
        x_50k = (x_50k_ph - 0.5) * 2.0

    with tf.name_scope('gan'):
        z = tf.random_normal([tf.shape(x_true)[0], 128], name="z")

        x_generated = generator(z, reuse=False)

        d_true = discriminator_fn(x_true, reuse=False)
        d_generated = discriminator_fn(x_generated, reuse=True)

        z_gen = tf.random_normal([args.batch_size * 2, 128], name="z")
        d_generated_train = discriminator_fn(generator(z_gen, reuse=True), reuse=True)

    with tf.name_scope('regularizer'):
        epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x_generated + (1 - epsilon) * x_true
        d_hat = discriminator_fn(x_hat, reuse=True)

        gradients = tf.gradients(d_hat, x_hat)[0]

        gradient_norms = stable_norm(gradients)

        gp = tf.maximum(gradient_norms - args.K, 0)
        gp_loss = args.lambda_gp * tf.reduce_mean(gp ** 2)

    with tf.name_scope('lipschitz_term'):
        max_lip_quotient = tf.reduce_max(gradient_norms)
        alpha = args.alpha if args.alpha > 1.0 else 1.0 + 1e-10
        multiplier = (
            1
            if alpha == float("inf") else
            (alpha - 1) / alpha
        )
        exponent = (
            1
            if alpha == float("inf") else
            alpha / (alpha - 1)
        )
        beta = (
            tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / args.iterations)) * (
                args.init_beta - args.final_beta
            ) + args.final_beta
        )
        lipschitz_term = multiplier * (beta * max_lip_quotient) ** exponent

    with tf.name_scope('loss_gan'):
        if args.forward:
            if args.tight:
                divergence = tf.reduce_mean(d_generated) - (
                    tf.reduce_mean(d_true) + conjugate_fn(d_true - tf.reduce_mean(d_true))
                )
            else:
                divergence = tf.reduce_mean(d_generated) - conjugate_fn(d_true)
        else:
            if args.tight:
                if args.penalized:
                    divergence = tf.reduce_mean(d_true) - (
                        tf.reduce_mean(d_generated) + conjugate_fn(d_generated - tf.reduce_mean(d_generated))
                    )
                else:
                    divergence = tf.reduce_mean(d_true) - conjugate_fn(d_generated)
            else:
                divergence = tf.reduce_mean(d_true) - conjugate_fn(d_generated)

        if args.alpha > 1.0 and args.tight:
            divergence -= lipschitz_term

        if args.forward:
            g_loss = tf.reduce_mean(d_generated_train)
        else:
            if args.penalized and args.tight:
                g_loss = -tf.reduce_mean(d_generated_train)
            else:
                g_loss = -conjugate_fn(d_generated_train)

        d_loss = -divergence
        if args.alpha == 1.0 and args.tight:
            d_loss += gp_loss

    with tf.name_scope('optimizer'):
        ema = EMAHelper(decay=args.ema_decay)

        decay = (
            tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / args.iterations))
            if args.decay_lr else
            1.0
        )
        learning_rate = args.lr * decay
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        with tf.control_dependencies(update_ops):
            g_train = optimizer.minimize(g_loss, var_list=g_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        with tf.control_dependencies(update_ops):
            d_train = optimizer.minimize(d_loss, var_list=d_vars)

    with tf.name_scope('summaries'):
        tf.summary.scalar('divergence', divergence)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)
        scalars_summary('d_true', d_true)
        scalars_summary('d_generated', d_generated)
        tf.summary.scalar('gp_loss', gp_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('global_step', global_step)
        scalars_summary('gradient_norms', gradient_norms)
        tf.summary.scalar('lipschitz_term', lipschitz_term)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)

        merged_summary = tf.summary.merge_all()

        # Advanced metrics
        with tf.name_scope('validation'):
            # INCEPTION VALIDATION
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)

            # Fixed z for fairness between runs
            inception_z = tf.constant(np.random.randn(10000, 128), dtype='float32')
            inception_score = tf.contrib.gan.eval.classifier_score(
                inception_z,
                classifier_fn=generate_resize_and_classify,
                num_batches=10000 // args.val_size
            )

            inception_summary = tf.summary.merge([
                tf.summary.scalar('inception_score', inception_score)
            ])

            full_summary = tf.summary.merge([
                tf.summary.image('x_true', image_grid(x_true)),
                tf.summary.image('x_generated', image_grid(x_generated)),
                tf.summary.image('gradients', image_grid(gradients)),
                tf.summary.image('gradients_true', image_grid(tf.gradients(d_true, x_true)[0])),
                tf.summary.image('gradients_generated', image_grid(tf.gradients(d_generated, x_generated)[0])),
                inception_summary,
            ])

        # Final eval
        with tf.name_scope('test'):
            # INCEPTION TEST
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)


            # Fixed z for fairness between runs
            inception_z_final = tf.constant(np.random.randn(100000, 128), dtype='float32')
            inception_score_final = tf.contrib.gan.eval.classifier_score(
                inception_z_final,
                classifier_fn=generate_resize_and_classify,
                num_batches=100000 // args.val_size
            )

            inception_summary_final = tf.summary.merge([
                tf.summary.scalar('inception_score_final', inception_score_final)
            ])

            # FID TEST
            def resize_and_classify(x):
                INCEPTION_FINAL_POOL = 'pool_3:0'
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_FINAL_POOL)

            fid_real_final = x_50k
            fid_z_final = tf.constant(np.random.randn(50000, 128), dtype='float32')
            fid_z_final_list = array_ops.split(fid_z_final, num_or_size_splits=50000 // args.val_size)
            fid_z_final_batches = array_ops.stack(fid_z_final_list)
            fid_gen_final = tf.map_fn(
                fn=partial(generator, reuse=True),
                elems=fid_z_final_batches,
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunGenerator'
            )
            fid_gen_final = array_ops.concat(array_ops.unstack(fid_gen_final), 0)
            fid_final = tf.contrib.gan.eval.frechet_classifier_distance(
                fid_real_final,
                fid_gen_final,
                classifier_fn=resize_and_classify,
                num_batches=50000 // args.val_size
            )

            fid_summary_final = tf.summary.merge([
                tf.summary.scalar('fid_final', fid_final)
            ])

            final_summary = tf.summary.merge([merged_summary, inception_summary_final, fid_summary_final])

        summary_writer = tf.summary.FileWriter(log_dir)

    # Initialize all TF variables
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Standardized validation z
    z_validate = np.random.randn(args.val_size, 128)

    print(f"Logging to: {log_dir}")

    # Train the network
    t = tqdm(range(args.iterations))
    for _ in t:
        i = sess.run(global_step)

        for j in range(args.n_critic):
            _, d_loss_result = sess.run(
                [d_train, d_loss],
                feed_dict={is_training: True}
            )

        _, g_loss_result, _ = sess.run(
            [g_train, g_loss, ema.apply],
            feed_dict={is_training: True}
        )

        if i % args.log_freq == args.log_freq - 1:
            merged_summary_result_train = sess.run(
                merged_summary,
                feed_dict={is_training: False}
            )
            summary_writer.add_summary(merged_summary_result_train, i)
        if i % args.val_freq == args.val_freq - 1:
            try:
                ema_dict = ema.average_dict()
                merged_summary_result_test = sess.run(
                    full_summary,
                    feed_dict={is_training: False, **ema_dict}
                )
                summary_writer.add_summary(merged_summary_result_test, i)
            except:
                print("Error during validation")

        t.set_description(
            f"[D loss: {d_loss_result:.3f}] [G loss: {g_loss_result:.3f}]]"
        )

        if (i + 1) == args.iterations:
            try:
                ema_dict = ema.average_dict()
                merged_summary_result_final = sess.run(
                    final_summary,
                    feed_dict={is_training: False, **ema_dict}
                )
                summary_writer.add_summary(merged_summary_result_final, args.iterations)
            except:
                print("Error during test")

