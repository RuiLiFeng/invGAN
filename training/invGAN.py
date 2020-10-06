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
# from training.iconv2d.conv2d_bijectors import invertible_conv2D_emerging as invConv2D
# from training.iconv2d.conv2d_bijectors import fast_inv_conv2d as invConv2D
from training.iconv2d.fourier import fast_fourier_conv as invConv2D


# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------
# Naive upsampling (nearest neighbor) and downsampling (average pooling).

def naive_upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H, 1, W, 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        return tf.reshape(x, [-1, C, H * factor, W * factor])

def naive_downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H // factor, factor, W // factor, factor])
        return tf.reduce_mean(x, axis=[3,5])

#----------------------------------------------------------------------------
# Modulated convolution layer.

def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
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
# Invertible Up and sampling
def Inv_UpSample(name, x, scale=False, reverse=False):
    """
    Upsample the given inputs by factor 2x2, but will decrease the channel num by factor of 4, such that
    the overall number of units remains unchanged
    :param name: name scope
    :param x: given inputs, [NHWC]
    :param scale: if true, scale the spatial size by factor of 2
    :param reverse: up sampling or down sampling
    :return: output tensor, [NHWC]
    """
    with tf.variable_scope(name):
        if not reverse:
            if scale:
                with tf.variable_scope('ConvScale'):
                    logdet1 = tf.zeros_like(x)[:, 0, 0, 0]
                    x1, _ = invConv2D('invConv1', x, logdet1, reverse=False)
                    x2, _ = invConv2D('invConv2', x, logdet1, reverse=False)
                    x3, _ = invConv2D('invConv3', x, logdet1, reverse=False)
                    x4, _ = invConv2D('invConv4', x, logdet1, reverse=False)
                    x = tf.concat([x1, x2, x3, x4], axis=3)
            x = upreshape(x)
            logdet = tf.zeros_like(x)[:, 0, 0, 0]
            x, _ = invConv2D('invConv', x, logdet, reverse=False)
        else:
            logdet = tf.zeros_like(x)[:, 0, 0, 0]
            x, _ = invConv2D('invConv', x, logdet, reverse=True)
            x = downshape(x)
            if scale:
                with tf.variable_scope('ConvScale'):
                    logdet = tf.zeros_like(x)[:, 0, 0, 0]
                    x, _ = invConv2D('invConv1', x[:, :, :, :x.shape[3].value // 4], logdet, reverse=True)
        return x


def upreshape(x): # [NHWC]
    x = tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value * 2, x.shape[3].value // 2])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value * 2, x.shape[3].value // 2])
    x = tf.transpose(x, [0, 2, 1, 3])
    return x


def downshape(x): # [NHWC], the inverse op of upreshape
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value // 2, x.shape[3].value * 2])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value // 2, x.shape[3].value * 2])
    return x


#----------------------------------------------------------------------------
# Invert scale conv2d in quotient space
def downscale_conv2d_layer(name, x, factor, reverse=False):
    """
    conv2d layer who downscale the channel of x, but keeps spatial size.
    Inverse map will choose an elements from the equivalent class of x,
    maintains an invert map in the quotient space.
    :param name: name scope
    :param x: input feature, [NHWC]
    :param factor: output channel nums decay factor
    :param reverse: whether compute reverse
    :return:
    """
    with tf.variable_scope(name):
        if not reverse:
            assert x.shape[3].value % factor == 0, "factor %d of downscale conv2d layer must" \
                                                  " be factor of input channel %d!" % (factor, x.shape[3].value)
            logdet = tf.zeros_like(x)[:, 0, 0, 0]
            x, _ = invConv2D('downscaleConv', x, logdet, reverse=False)
            xs = tf.split(x, factor, axis=3)
            x = tf.math.add_n(xs) / factor
        else:
            x = tf.concat([x for _ in range(factor)], axis=3)
            logdet = tf.zeros_like(x)[:, 0, 0, 0]
            x, _ = invConv2D('downscaleConv', x, logdet, reverse=True)
        return x


#----------------------------------------------------------------------------
def inv_toRGB(name, x, fin, reverse=False):
    """
    ToRGB op with inverse in the quotient space
    :param name: name scope
    :param x: input [NHWC]
    :param fin: original input channel num
    :param reverse:
    :return:
    """
    with tf.variable_scope(name):
        if fin == 3:
            if not reverse:
                assert fin == x.shape[3].value
                logdet = tf.zeros_like(x)[:, 0, 0, 0]
                x, _ = invConv2D('channel_shuffle', x, logdet, ksize=3, reverse=False)
                x, _ = invConv2D('toRGB', x, logdet, ksize=1, reverse=False, use_fourier_forward=True)

            else:
                logdet = tf.zeros_like(x)[:, 0, 0, 0]
                x, _ = invConv2D('toRGB', x, logdet, ksize=1, reverse=True, use_fourier_forward=True)
                x, _ = invConv2D('channel_shuffle', x, logdet, ksize=3, reverse=True)
        else:
            if not reverse:
                assert fin == x.shape[3].value
                logdet = tf.zeros_like(x)[:, 0, 0, 0]
                x, _ = invConv2D('channel_shuffle', x, logdet, ksize=3, reverse=False)
                x = x[:, :, :, :-(x.shape[3].value % 3)]
                xs = tf.split(x, x.shape[3].value // 3, axis=3)
                x = tf.math.add_n(xs) / (x.shape[3].value // 3)
                x, _ = invConv2D('toRGB', x, logdet, ksize=1, reverse=False, use_fourier_forward=True)

            else:
                logdet = tf.zeros_like(x)[:, 0, 0, 0]
                x, _ = invConv2D('toRGB', x, logdet, ksize=1, reverse=True, use_fourier_forward=True)
                xs = [x for _ in range(fin // 3)] + [x[:, :, :, :fin % 3]]
                x = tf.concat(xs, axis=3)
                x, _ = invConv2D('channel_shuffle', x, logdet, ksize=3, reverse=True)
        return x


#----------------------------------------------------------------------------
# Inv Module Conv
def inv_module_conv2d_layer(x, up=False, downscale=False, reverse=False):
    if up:
        x = Inv_UpSample('invupdown', x, reverse=reverse)
    elif downscale:
        x = downscale_conv2d_layer('downConv', x, 2, reverse=reverse)
    else:
        logdet = tf.zeros_like(x)[:, 0, 0, 0]
        x, _ = invConv2D('invConv', x, logdet, reverse=reverse)
    return x


#----------------------------------------------------------------------------
# Invert Dense layer
def inv_dense_layer(x, fmaps, gain=1, use_wscale=True,
                    lrmul=1, weight_var='weight', reverse=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    assert x.shape[1].value == fmaps
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)

    if reverse:
        w = tf.matrix_inverse(w)

    return tf.matmul(x, w)


#----------------------------------------------------------------------------
# Invert bias act
def inv_bias_act(x, act='linear', alpha=0.2, gain=None, lrmul=1, bias_var='bias', reverse=False):
    assert act in ['linear', 'lrelu']
    x = tf.transpose(x, [0, 3, 1, 2])
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    if reverse:
        if act == 'lrelu':
            if gain is None:
                gain = np.sqrt(2)
            x = x / gain
            mask = tf.cast(x < 0, x.dtype) * (1.0/alpha - 1.0) + 1.0
            x = x * mask
        x = fused_bias_act(x, -b)
        return tf.transpose(x, [0, 2, 3, 1])
    return tf.transpose(fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain), [0, 2, 3, 1])


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
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
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

    # Normalize latents.
    if normalize_latents:
        with tf.variable_scope('Normalize'):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# InvGAN without low rank decomposition
def G_quotient(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
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
    fused_modconv       = True,
    fmap_final          = 4,
    use_noise           = False,
    report_layer        = False,
    **_kwargs):

    assert dlatent_size == fmap_final * resolution * resolution and dlatent_size % 16 == 0, "dlatent_size %d," \
                                                                                            "fmap_final %d, " \
                                                                                            "resolution %d," \
                                                                                            % (dlatent_size,
                                                                                               fmap_final,
                                                                                               resolution)
    assert num_channels == 3
    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    num_layers = resolution_log2 * 2 - 2
    images_out = None
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    latents_in = dlatents_in[:, 0, :]


    # Primary inputs
    # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2 ** res, 2 ** res]
        noise_inputs.append(
            tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    layer_dict = {}

    # Single Conv layer
    def layer(x, layer_idx, up=False):
        # downscale = (layer_idx >= 8)
        downscale = False
        x = inv_module_conv2d_layer(x, up, downscale)
        if use_noise:
            if randomize_noise:
                noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            else:
                noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        x = inv_bias_act(x, act=act)
        return x

    # Early layers.
    with tf.variable_scope('4x4'):
        x = tf.reshape(latents_in, [-1, 4, 4, dlatent_size // 16])
        layer_dict.update({'layer4x4': x})
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0)

    def block(res, x):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                x = layer(x, layer_idx=res*2-5, up=True)
            with tf.variable_scope('Conv1'):
                x = layer(x, layer_idx=res*2-4, up=False)
            return x

    def torgb(res, x):
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            x = inv_toRGB('Conv', x, x.shape[3].value)
            x = inv_bias_act(x)
            return x

    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(res, x)
            layer_dict.update({'block%dx%d'% (2**res, 2**res): x})
            if res == resolution_log2:
                x = torgb(res, x)
                layer_dict.update({'torgb': x})

    images_out = tf.transpose(x, [0, 3, 1, 2])

    assert images_out.dtype == tf.as_dtype(dtype)
    if report_layer:
        return tf.identity(images_out, name='images_out'), layer_dict
    else:
        return tf.identity(images_out, name='images_out')



def Q_infer(
        images_in,
        dlatent_size,
        resolution           = 64,
        num_channels         = 3,
        fmap_final           = 4,
        use_noise            = False,
        randomize_noise      = True,
        nonlinearity         = 'lrelu',
        dtype                = 'float32',
        **_kwargs):
    assert dlatent_size == fmap_final * resolution * resolution and dlatent_size % 16 == 0
    assert num_channels == 3
    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    num_layers = resolution_log2 * 2 - 2

    images_in = tf.transpose(images_in, [0, 2, 3, 1])
    images_in = tf.cast(images_in, dtype)

    noise_inputs = []
    # for layer_idx in range(num_layers - 1):
    #     res = (layer_idx + 5) // 2
    #     shape = [1, 1, 2 ** res, 2 ** res]
    #     noise_inputs.append(
    #         tf.get_variable('noise_inf%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(),
    #                         trainable=False))


    def inv_layer(x, layer_idx, up=False):
        # downscale = (layer_idx >= 8)
        x = inv_bias_act(x, act=act, reverse=True)
        if use_noise:
            if randomize_noise:
                noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            else:
                noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)

        downscale = False
        x = inv_module_conv2d_layer(x, up, downscale, reverse=True)
        return x
    def inv_block(res, x):
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            with tf.variable_scope('Conv1'):
                x = inv_layer(x, layer_idx=res * 2 - 4, up=False)
            with tf.variable_scope('Conv0_up'):
                x = inv_layer(x, layer_idx=res * 2 - 5, up=True)
            return x
    def inv_torgb(res, x):
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            x = inv_bias_act(x, reverse=True)
            x = inv_toRGB('Conv', x, fmap_final, reverse=True)
            return x

    x = images_in

    layer_dict = {}
    for res in range(resolution_log2, 3 - 1, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == resolution_log2:
                x = inv_torgb(res, x)
                layer_dict.update({'inv_torgb': x})
            x = inv_block(res, x)
            layer_dict.update({'inv_block%dx%d'% (2**res, 2**res): x})

        # Early layers.
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Conv'):
            x = inv_layer(x, layer_idx=0)
            layer_dict.update({'inv_layer4x4': x})
        latents = tf.reshape(x, [-1, np.prod(x.shape[1:])])

    assert latents.dtype == tf.as_dtype(dtype)
    return tf.identity(latents, name='latents_infered'), layer_dict


#----------------------------------------------------------------------------
# Original StyleGAN discriminator.
# Used in configs B-D (Table 1).

def D_stylegan(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks for spatial layers.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=1), act=act)
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0'):
                x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
            with tf.variable_scope('Conv1_down'):
                x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            with tf.variable_scope('Downsample_lod%d' % lod):
                img = downsample_2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = tflib.lerp_clip(x, y, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            x = lambda: fromrgb(naive_downsample_2d(images_in, factor=2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromrgb(naive_downsample_2d(images_in, factor=2**(lod+1)), res - 1), lod_in - lod))
            return y()
        x = grow(3, resolution_log2 - 3)

    # Final layers at 4x4 resolution.
    with tf.variable_scope('4x4'):
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

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
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------


"""
from training.invGAN import *
f = G_quotient
q= Q_infer
d = 3
resolution=64
z = tf.random.normal([8,4096*d])
with tf.variable_scope('test',reuse=tf.AUTO_REUSE):
    x =G_quotient(z,4096*d,fmap_final=d, resolution=resolution)
    z1 =q(x, 4096*d, fmap_final=d, resolution=resolution)
    x1 = G_quotient(z1,4096*d,fmap_final=d, resolution=resolution)

def err(a,b):return tf.reduce_sum(tf.square(a-b))


e = err(x,x1)
init = tf.global_variables_initializer
sess = tf.Session()
sess.run(init())
sess.run(e)


def inv_layer(x, layer_idx, up=False):
    # downscale = (layer_idx >= 8)
    x = inv_bias_act(x, act=act, reverse=True)
    if use_noise:
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
    downscale = False
    x = inv_module_conv2d_layer(x, up, downscale, reverse=True)
    return x
def inv_block(res, x):
    with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
        with tf.variable_scope('Conv1'):
            x = inv_layer(x, layer_idx=res * 2 - 4, up=False)
        with tf.variable_scope('Conv0_up'):
            x = inv_layer(x, layer_idx=res * 2 - 5, up=True)
        return x
def inv_torgb(res, x):
    lod = resolution_log2 - res
    with tf.variable_scope('ToRGB_lod%d' % lod):
        x = inv_bias_act(x, reverse=True)
        x = inv_toRGB('Conv', x, fmap_final, reverse=True)
        return x



def layer(x, layer_idx, up=False):
    # downscale = (layer_idx >= 8)
    downscale = False
    x = inv_module_conv2d_layer(x, up, downscale)
    if use_noise:
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
    x = inv_bias_act(x, act=act)
    return x


def block(res, x):
    with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res * 2 - 5, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res * 2 - 4, up=False)
        return x


def torgb(res, x):
    lod = resolution_log2 - res
    with tf.variable_scope('ToRGB_lod%d' % lod):
        x = inv_toRGB('Conv', x, x.shape[3].value)
        x = inv_bias_act(x)
        return x
        
fmap_final=32
use_noise=False
act='lrelu'
resolution_log2=6
with tf.variable_scope('tt',reuse=tf.AUTO_REUSE):
    x = tf.random.normal([12,32,32,128])
    y = block(2,x)
    z = torgb(2,y)
    y1 = inv_torgb(2,z)
    x1 = inv_block(2,y1)
    y2 = block(2,x1)
    z1 = torgb(2,y2)
"""