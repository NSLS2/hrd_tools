import numpy as np
import xrt.backends.raycing.sources as rsources


def pattern_sample(tth, cumsum, N):
    return np.arccos(np.sin(tth)[np.searchsorted(cumsum, np.random.rand(N))])


class XrdSource(rsources.GeometricSource):
    def __init__(
        self,
        *args,
        pattern,
        # in Deg
        vertical_divergence: float = 0,
        # in Deg
        horizontal_divergence: float = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pattern = pattern
        self._vertical_diivergence = vertical_divergence
        self._horizontal_diivergence = horizontal_divergence

    def _set_annulus(
        self,
        axis1,
        axis2,
        rMin,
        rMax,
        phiMin,
        phiMax,
    ):
        # if rMax > rMin:
        #    A = 2. / (rMax**2 - rMin**2)
        #    r = np.sqrt(2*np.random.uniform(0, 1, self.nrays)/A + rMin**2)
        # else:
        #    r = rMax

        full_tth = self._pattern.theta.to_numpy()
        start_indx, stop_indx = np.searchsorted(full_tth, [rMin, rMax])
        trimmed_pattern = self._pattern.I1.to_numpy()[start_indx:stop_indx]
        trimmed_tth = full_tth[start_indx:stop_indx]
        # print(f"{rMin=}, {rMax=}, {start_indx=}, {stop_indx=}")
        # print(len(full_tth))
        # print(len(trimmed_tth))

        I_cumsum = trimmed_pattern.cumsum()
        I_cumsum /= I_cumsum[-1]

        # TODO trim tth/r
        tth = trimmed_tth[np.searchsorted(I_cumsum, np.random.rand(self.nrays))]
        # print(f"{tth.min()=}, {tth.max()=}")
        # TODO only generate extra random numbers if needed
        if self._vertical_diivergence > 0:
            extra_vertical = self._vertical_diivergence * np.random.rand(self.nrays)
        else:
            extra_vertical = 0

        if self._horizontal_diivergence > 0:
            extra_horizontal = self._horizontal_diivergence * np.random.rand(self.nrays)
        else:
            extra_horizontal = 0

        a = np.tan(np.deg2rad(tth + extra_vertical))
        b = np.tan(np.deg2rad(tth + extra_horizontal))
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = b * np.cos(phi)
        axis2[:] = a * np.sin(phi)


# because xrt.glow has a regex on the str of the type
XrdSource.__module__ = rsources.GeometricSource.__module__
