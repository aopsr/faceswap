import numpy as np
from ._base import Adjustment
from scipy import stats

class Color(Adjustment):
    def process(self, source1, target1, raw_mask, bins=255, n_iter=4) -> np.ndarray:
        """Iterative Disribution Transfer
        https://github.com/ptallada/colour_transfer

        """
        reference = source1 * raw_mask
        target = target1 * raw_mask

        shape = target.shape
        n_dims = shape[-1]

        target = target.reshape(-1, 3)
        reference = reference.reshape(-1, 3)

        for _ in range(n_iter):
            r = stats.special_ortho_group.rvs(n_dims)

            d0r = r @ target.T
            d1r = r @ reference.T
            d_r = np.empty_like(target.T)

            for j in range(n_dims):
                lo = min(d0r[j].min(), d1r[j].min())
                hi = max(d0r[j].max(), d1r[j].max())

                p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
                p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

                cp0r = p0r.cumsum().astype(float)
                cp0r /= cp0r[-1]

                cp1r = p1r.cumsum().astype(float)
                cp1r /= cp1r[-1]

                f = np.interp(cp0r, cp1r, edges[1:])

                d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

            target = np.linalg.solve(r, (d_r - d0r)).T + target

        output = target.reshape(shape)

        return output