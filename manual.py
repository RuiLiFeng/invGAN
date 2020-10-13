import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from training import misc

import pretrained_networks
from training.invGAN import *
from  training.iconv2d.fourier import compute_logdet

network_pkl = '/gdata2/fengrl/invGAN/00002-InvGan-ffhq-64-4gpu-config-e/network-snapshot-011040.pkl'
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)

f = G_quotient
q= Q_infer
d = 3
resolution=64
# z = tf.random.normal([8,4096*d])
# w = tf.tile(z[:,np.newaxis],[1,10,1])
z = tf.random.normal([8,512])
w = Gs.components.mapping.get_output_for(z,None)
with tf.variable_scope('G_synthesis_1',reuse=True):
    x =G_quotient(w,4096*d,fmap_final=d, resolution=resolution)
    z1,_ =q(x, 4096*d, fmap_final=d, resolution=resolution)
    w1 = tf.tile(z1[:,np.newaxis],[1,10,1])
    x1 = G_quotient(w1,4096*d,fmap_final=d, resolution=resolution)

def err(a,b):return tf.reduce_sum(tf.square(a-b))



def abserr(a,b): return err(a,b)/err(a,0)



e = err(x,x1)
# init = tf.global_variables_initializer
# sess = tf.Session()
# sess.run(init())
# sess.run(e)
# sess.run(abserr(x,x1))
tflib.run(e)
tflib.run(abserr(x, x1))

# img, img_re = sess.run([x, x1])
img, img_re = tflib.run([x, x1])

misc.save_image_grid(img,
                         dnnlib.make_run_dir_path('/gdata2/fengrl/img.png'), drange=[-1, 1], grid_size=[8, 1])

misc.save_image_grid(img_re,
                         dnnlib.make_run_dir_path('/gdata2/fengrl/img_re.png'), drange=[-1, 1], grid_size=[8, 1])


v = [v for v in tf.global_variables() if 'weight' in v.name]
w = v[4]
ww = v[4] * 5.0
f_shape = [16,16]
d = tf.rsqrt(tf.reduce_sum(tf.square(w), axis=[0, 1, 2]) + 1e-8)
dd = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[0, 1, 2]) + 1e-8)
w *= d[np.newaxis, np.newaxis, np.newaxis, :]

w_fft = tf.spectral.rfft2d(
    tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
    fft_length=f_shape,
    name=None
)
dlogdet = compute_logdet(w_fft, 16)


w_fft_inv = tf.linalg.inv(
    tf.transpose(w_fft, [2, 3, 0, 1]),
)

ww_fft = tf.spectral.rfft2d(
    tf.transpose(ww, [3, 2, 0, 1])[:, :, ::-1, ::-1],
    fft_length=f_shape,
    name=None
)


ww_fft_inv = tf.linalg.inv(
    tf.transpose(ww_fft, [2, 3, 0, 1]),
)


# Dimension [c_in, c_out, v, u]
w_fft_inv = tf.transpose(w_fft_inv, [2, 3, 0, 1])
ww_fft_inv = tf.transpose(ww_fft_inv, [2, 3, 0, 1])



w_,d_,wft,wft_inv, ww_, dd_, wwft, wwft_inv = tflib.run([w, d, w_fft, w_fft_inv, ww, dd, ww_fft, ww_fft_inv])


norm = np.linalg.norm

def reg_(r=0.0, sigma=0.1):
    # Dimension [c_out, c_in, v, u]
    w_fft = tf.spectral.rfft2d(
        tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
        fft_length=f_shape,
        name=None
    )
    # Dimension [v, u, c_in, c_out], channels switched because of
    # inverse.
    w_fft = tf.transpose(w_fft, [2, 3, 0, 1])
    noise_re = tf.random.normal([64, w_fft.shape[-1].value])
    noise_im = tf.random.normal([64, w_fft.shape[-1].value])
    noise = tf.complex(noise_re, noise_im)
    noise *= tf.complex(tf.rsqrt(tf.reduce_sum(tf.square(tf.math.abs(noise)), axis=1, keepdims=True)), 0.0)
    noise_transform = tf.matmul(w_fft, tf.transpose(noise, [1, 0]))
    transform_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.math.abs(noise_transform)), axis=2))
    pelnety = tf.reduce_sum(tf.nn.relu(sigma - transform_norm))
    return r + pelnety


w_fft = tf.spectral.rfft2d(
        tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
        fft_length=f_shape,
        name=None
    )
# Dimension [v, u, c_in, c_out], channels switched because of
# inverse.
w_fft = tf.transpose(w_fft, [2, 3, 0, 1])
noise_re = tf.random.normal([64, w_fft.shape[-1].value])
noise_im = tf.random.normal([64, w_fft.shape[-1].value])
noise = tf.complex(noise_re, noise_im)
noise *= tf.complex(tf.rsqrt(tf.reduce_sum(tf.square(tf.math.abs(noise)), axis=1, keepdims=True)), 0.0)
noise_transform = tf.matmul(w_fft, tf.transpose(noise, [1, 0]))
transform_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.math.abs(noise_transform)), axis=2))
pelnety = tf.reduce_sum(tf.nn.relu(sigma - transform_norm))