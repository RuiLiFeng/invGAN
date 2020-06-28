# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Network architectures used in the StyleGAN2 paper."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.
#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

def G_main(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.5,                      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                      # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',              # Build func name for the mapping network.
    synthesis_func          = 'G_synthesis_stylegan2',  # Build func name for the synthesis network.
    **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out = components.synthesis.get_output_for(dlatents, is_training=is_training, force_clean_graph=is_template_graph, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out

#----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).

def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 100,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 100,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    act = mapping_nonlinearity

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')



def G_synthesis_stylegan2(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 100,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    noise_style         = None,
    clip_style          = 'ffhq',
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    act = tf.nn.relu
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)


    # Noise inputs.
    noise_inputs = []
    for res in range(2, 6):
        shape = [1, 1, 2 ** res, 2 ** res]
        noise_inputs.append(
            tf.get_variable('noise%d' % res, shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    with tf.variable_scope('4x4'):
        x = dlatents_in[:, 0]
        x = tf.layers.dense(x, 32 * 8 * 4, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        x = tf.layers.batch_normalization(x, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = tf.reshape(x, [-1, 32*8, 2, 2])
        x = act(x)
        x = add_noise(x, noise_style, randomize_noise, noise_inputs, 0, clip_style)

    with tf.variable_scope('8x8'):
        x = tf.layers.conv2d_transpose(x, 32 * 4, 5, 2, padding='same', use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = act(x)
        x = add_noise(x, noise_style, randomize_noise, noise_inputs, 1, clip_style)

    with tf.variable_scope('16x16'):
        x = tf.layers.conv2d_transpose(x, 32 * 2, 5, 2, padding='same', use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = act(x)
        x = add_noise(x, noise_style, randomize_noise, noise_inputs, 2, clip_style)

    with tf.variable_scope('32x32'):
        x = tf.layers.conv2d_transpose(x, 32 * 1, 5, 2, padding='same', use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = act(x)
        x = add_noise(x, noise_style, randomize_noise, noise_inputs, 3, clip_style)

    with tf.variable_scope('64x64'):
        x = tf.layers.conv2d_transpose(x, 3, 5, 2, padding='same', use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = tf.nn.tanh(x)

    images_out = x
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')



#----------------------------------------------------------------------------
# StyleGAN2 discriminator (Figure 7).
# Implements skip connections and residual nets (Figure 7), but no progressive growing.
# Used in configs E-F (Table 1).

def D_stylegan2(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    with tf.variable_scope('64x64'):
        x = tf.layers.conv2d(images_in, 32, 5, 2, padding='same', use_bias=False,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format='channels_first')
        x = act(x)

    with tf.variable_scope('32x32'):
        x = tf.layers.conv2d(x, 32 * 2, 5, 2, padding='same', use_bias=False,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        x = act(x)

    with tf.variable_scope('16x16'):
        x = tf.layers.conv2d(x, 32 * 4, 5, 2, padding='same', use_bias=False,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))

        x = act(x)

    with tf.variable_scope('8x8'):
        x = tf.layers.conv2d(x, 32 * 8, 5, 2, padding='same', use_bias=False,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=1, gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))

        x = act(x)

    with tf.variable_scope('4x4'):
        x = tf.layers.dense(tf.reshape(x, [-1, np.prod(x.shape[1:])]), 1,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    scores_out = x
    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x


def adjust_range(x):
    assert len(x.shape) == 4
    with tf.variable_scope('Adjust_range'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        x_max = tf.reduce_max(x, axis=(2, 3), keepdims=True)
        x = x / (x_max + 1e-8)
        x = tf.cast(x, orig_dtype)
        return x


def spatial_att(x, clip_style):
    """
    Spatial attention mask
    :param x: [NCHW]
    :return: None negative mask tensor [NCHW]
    """
    fmaps = x.shape[1].value
    if clip_style == 'ffhq':
        x = tf.reduce_sum(tf.nn.relu(-x), axis=1, keepdims=True)
    elif clip_style == 'cat':
        x = tf.reduce_sum(x, axis=1, keepdims=True)
    elif clip_style == 'church':
        x = tf.reduce_max(-x, axis=1, keepdims=True)
    else:
        raise ValueError('Unsupported clip style %s' % clip_style)
    x = (adjust_range(x) + 1.0) / 2.0
    with tf.variable_scope('x_mask'):
        x_mask = get_weight(shape=[x.shape[2].value, x.shape[3].value])
    with tf.variable_scope('bias'):
        b = get_weight(shape=[x.shape[2].value, x.shape[3].value])
    att = x * x_mask + b
    return tf.tile(att, [1, fmaps, 1, 1])


def add_noise(x, noise_style, randomize_noise, noise_inputs, layer_idx, clip_style):
    if noise_style is not None:
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        if noise_style == 'add':
            x += noise * tf.cast(noise_strength, x.dtype)
        elif noise_style == 're':
            with tf.variable_scope('resampling'):
                alpha = tf.get_variable('alpha', shape=[], initializer=tf.initializers.constant(0.5))
                sp_att_mask = alpha + (1 - alpha) * spatial_att(x, clip_style)
                sp_att_mask *= tf.rsqrt(tf.reduce_mean(tf.square(sp_att_mask), axis=[2, 3], keepdims=True) + 1e-8)
                x += noise * tf.cast(noise_strength, x.dtype)
                x = x * sp_att_mask
    return x