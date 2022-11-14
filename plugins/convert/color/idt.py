import numpy as np
from ._base import Adjustment
from scipy import stats
import cv2

class Color(Adjustment):
    def process(self, source, target, raw_mask, bins=300, n_rot=10, relaxation=1):
        """Iterative Disribution Transfer
        https://github.com/ptallada/colour_transfer

        """
        i0 = target.copy()
        i1 = source.copy()

        i0[raw_mask[..., 0] == 0] = [0,0,0]
        i1[raw_mask[..., 0] == 0] = [0,0,0]

        i0 = i0.reshape(-1, 3)
        i1 = i1.reshape(-1, 3)

        n_dims = i0.shape[-1]
        
        d0 = i0.T
        d1 = i1.T
        
        for _ in range(n_rot):
            
            r = stats.special_ortho_group.rvs(n_dims).astype(np.float32)
            
            d0r = np.dot(r, d0)
            d1r = np.dot(r, d1)
            d_r = np.empty_like(d0)
            
            for j in range(n_dims):
                
                lo = min(d0r[j].min(), d1r[j].min())
                hi = max(d0r[j].max(), d1r[j].max())
                
                p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
                p1r, _     = np.histogram(d1r[j], bins=bins, range=[lo, hi])

                cp0r = p0r.cumsum().astype(np.float32)
                cp0r /= cp0r[-1]

                cp1r = p1r.cumsum().astype(np.float32)
                cp1r /= cp1r[-1]
                
                f = np.interp(cp0r, cp1r, edges[1:])
                
                d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)
            
            d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0
        
        return np.clip(d0.T.reshape(target.shape), 0, 1) # TODO: regrain

    
    def regrain(self, i0, ir, varargin):
        pass