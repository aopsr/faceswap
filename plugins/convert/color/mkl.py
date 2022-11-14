import numpy as np
import cv2
from ._base import Adjustment

class Color(Adjustment):
    def process(self, source, target, raw_mask) -> np.ndarray:
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
        return np.clip (result.reshape(target.shape).astype(x0.dtype), 0, 1)