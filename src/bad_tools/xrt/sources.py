import numpy as np
import xrt.backends.raycing.sources as rsources


def pattern_sample(tth, cumsum, N):
    return np.arccos(np.sin(tth)[np.searchsorted(cumsum, np.random.rand(N))])


class XrdSource(rsources.GeometricSource):
    def __init__(self, *args, pattern, **kwargs):
        super().__init__(*args, **kwargs)
        self._pattern = pattern
        self._I_cumsum = (pattern.I1 / pattern.I1.sum()).cumsum()

    def _set_annulus(self, axis1, axis2, rMin, rMax, phiMin, phiMax):
        # if rMax > rMin:
        #    A = 2. / (rMax**2 - rMin**2)
        #    r = np.sqrt(2*np.random.uniform(0, 1, self.nrays)/A + rMin**2)
        # else:
        #    r = rMax
        I_cumsum = self._I_cumsum
        # TODO trim tth/r
        tth = self._pattern.theta[
            np.searchsorted(I_cumsum, np.random.rand(self.nrays))
        ].to_numpy()
        r = np.tan(np.deg2rad(tth))
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = r * np.cos(phi)
        axis2[:] = r * np.sin(phi)


# because xrt.glow has a regex on the str of the type
XrdSource.__module__ = rsources.GeometricSource.__module__
