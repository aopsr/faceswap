import numpy as np
import cv2
from ._base import Adjustment

class Color(Adjustment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._cuda:
            import tensorflow as tf
            import tensorflow_probability as tfp
            import tensorflow.linalg as ln
            def graph(x1, x0):
                output_shape = x0.shape
                eps = tf.math.nextafter(tf.constant(0, dtype=tf.float32),tf.constant(1, dtype=tf.float32))

                x0 = tf.reshape(x0, [-1,3])
                x1 = tf.reshape(x1, [-1,3])

                a = tfp.stats.covariance(x0)
                b = tfp.stats.covariance(x1)

                Da2, Ua = ln.eig(a)

                Ua = tf.math.real(Ua)

                Da = ln.diag(tf.math.sqrt(tf.math.maximum(tf.math.real(Da2), eps)))

                C = ln.matmul(ln.matmul(ln.matmul(ln.matmul(Da, Ua, transpose_b=True), b), Ua), Da)

                Dc2, Uc = ln.eig(C)
                Uc = tf.math.real(Uc)

                Dc = ln.diag(tf.math.sqrt(tf.math.maximum(tf.math.real(Dc2), eps)))

                Da_inv = ln.diag(tf.math.reciprocal(ln.diag_part(Da)))

                t = ln.matmul(ln.matmul(ln.matmul(ln.matmul(ln.matmul(ln.matmul(Ua, Da_inv), Uc), Dc), Uc, transpose_b=True), Da_inv), Ua, transpose_b=True)

                mx0 = tf.math.reduce_mean(x0, axis=0)
                mx1 = tf.math.reduce_mean(x1, axis=0)

                result = tf.math.add(ln.matmul(tf.math.subtract(x0, mx0), t), mx1)

                return tf.reshape(result, output_shape)
            self._process = tf.function(graph)

    def process(self, source, target, raw_mask) -> np.ndarray:
        if self._cuda:
            x0 = target.copy()
            x1 = source.copy()

            x0[raw_mask[..., 0] == 0] = [0,0,0]
            x1[raw_mask[..., 0] == 0] = [0,0,0]

            return self._process(x1, x0).numpy()

        else:
            eps = np.finfo(float).eps

            x0 = target.copy()
            x1 = source.copy()

            x0[raw_mask[..., 0] == 0] = [0,0,0]
            x1[raw_mask[..., 0] == 0] = [0,0,0]

            h,w,c = x0.shape
            h1,w1,c1 = x1.shape

            x0 = x0.reshape ( (h*w,c) )
            x1 = x1.reshape ( (h1*w1,c1) )

            a = np.cov(x0.T)
            b = np.cov(x1.T)

            Da2, Ua = np.linalg.eig(a)
            Da = np.diag(np.sqrt(Da2.clip(eps, None)))

            C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

            Dc2, Uc = np.linalg.eig(C)

            Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

            Da_inv = np.diag(1./(np.diag(Da)))

            t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

            mx0 = np.mean(x0, axis=0)
            mx1 = np.mean(x1, axis=0)

            result = np.dot(x0-mx0, t) + mx1

            return result.reshape(target.shape).astype(x0.dtype)
