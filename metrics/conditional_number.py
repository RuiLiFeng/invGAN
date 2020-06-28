# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
import dnnlib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class CondN(metric_base.MetricBase):
    def __init__(self, num_samples, epsilon, space, crop, minibatch_per_gpu, report_type, Gs_overrides, **kwargs):
        assert space in ['z', 'w']
        assert report_type in ['mean', 'max']
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.space = space
        self.crop = crop
        self.minibatch_per_gpu = minibatch_per_gpu
        self.report_type = report_type
        self.Gs_overrides = Gs_overrides

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        Gs_kwargs = dict(Gs_kwargs)
        Gs_kwargs.update(self.Gs_overrides)
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        cond_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if
                              name.startswith('noise')]
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                if self.space == 'w':
                    dlat = Gs_clone.components.mapping.get_output_for(latents, None, **Gs_kwargs)
                    dlat = tf.cast(dlat, tf.float32)
                    epi = tf.random_normal(dlat.shape, stddev=self.epsilon)
                    dlat_in = tf.concat([dlat, dlat + epi], axis=0)
                    x = dlat
                else: # space == 'z
                    epi = tf.random_normal(latents.shape, stddev=self.epsilon)
                    dlat = tf.concat([latents, latents + epi], axis=0)
                    dlat_in = Gs_clone.components.mapping.get_output_for(dlat, None, **Gs_kwargs)
                    x = latents
                with tf.control_dependencies(
                        [var.initializer for var in noise_vars]):  # use same noise inputs for the entire minibatch
                    images = Gs_clone.components.synthesis.get_output_for(dlat_in, randomize_noise=False, **Gs_kwargs)
                    images = tf.cast(images, tf.float32)

                    # Crop only the face region.
                if self.crop:
                    c = int(images.shape[2] // 8)
                    images = images[:, :, c * 3: c * 7, c * 2: c * 6]

                def norm(v): return tf.sqrt(tf.reduce_sum(tf.square(v), axis=list(range(1, len(v.shape)))))

                cond = (norm(images[:self.minibatch_per_gpu] - images[self.minibatch_per_gpu:]) /
                        norm(images[:self.minibatch_per_gpu])) / (norm(epi) / norm(x))
                cond_expr.append(cond)

        # Sampling loop
        all_cond = []
        for begin in range(0, self.num_samples, minibatch_size):
            self._report_progress(begin, self.num_samples)
            all_cond += tflib.run(cond_expr)
        all_cond = np.concatenate(all_cond, axis=0)
        all_cond.sort()
        # if self.report_type == 'mean':
        self._report_result((np.mean(all_cond), np.max(all_cond), np.mean(all_cond[-1000:])), fmt='%-10.4f    %-10.4f    %-10.4f')
        # else:
        #     self._report_result(np.max(all_cond))

#----------------------------------------------------------------------------
