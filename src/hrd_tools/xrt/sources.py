import numpy as np
import pandas as pd
import xrt.backends.raycing.sources as rsources
from xrt.backends.raycing.sources.beams import allArguments as _base_allArguments

# `xrt.backends.raycing._flow_utils.get_params` filters discovered __init__
# kwargs against the `allArguments` attribute on the *module* that owns the
# class.  We must therefore expose the union of GeometricSource's canonical
# kwargs and our additions here, otherwise glow/JSON round-trip silently drops
# every standard kwarg (center, nrays, dx, distxprime, ...).
allArguments = tuple(
    dict.fromkeys(
        _base_allArguments
        + ("pattern_path", "vertical_divergence", "horizontal_divergence")
    )
)


def pattern_sample(tth, cumsum, N):
    return np.arccos(np.sin(tth)[np.searchsorted(cumsum, np.random.rand(N))])


def _load_pattern(pattern_path):
    """Load a reference XRD pattern from disk.

    The expected format is a whitespace-separated text file with three
    columns (theta, I1, I0) and three header rows that are skipped.
    """
    return pd.read_csv(
        pattern_path,
        skiprows=3,
        names=["theta", "I1", "I0"],
        sep=" ",
        skipinitialspace=True,
        index_col=False,
    )


class XrdSource(rsources.GeometricSource):
    def __init__(
        self,
        *args,
        # NOTE: keep a default (even None) so xrt._flow_utils.get_params picks
        # this up via argspec.kwonlydefaults — args without defaults are
        # silently dropped from the glow/JSON kwargs.  We accept a path
        # (string) rather than an in-memory DataFrame so the source survives
        # JSON round-tripping through xrt.glow / xrt.export_to_json.
        pattern_path=None,
        # in Deg as FWHM
        vertical_divergence: float = 0,
        # in Deg as FWHM
        horizontal_divergence: float = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pattern_path = pattern_path
        self.pattern = _load_pattern(pattern_path) if pattern_path is not None else None
        self.vertical_divergence = vertical_divergence
        self.horizontal_divergence = horizontal_divergence
        self._rng = np.random.default_rng()

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

        full_tth = self.pattern.theta.to_numpy()
        start_indx, stop_indx = np.searchsorted(full_tth, [rMin, rMax])
        trimmed_pattern = self.pattern.I1.to_numpy()[start_indx:stop_indx]
        trimmed_tth = full_tth[start_indx:stop_indx]
        # print(f"{rMin=}, {rMax=}, {start_indx=}, {stop_indx=}")
        # print(len(full_tth))
        # print(len(trimmed_tth))

        I_cumsum = trimmed_pattern.cumsum()
        I_cumsum -= I_cumsum[0]
        I_cumsum /= I_cumsum[-1]

        tth = trimmed_tth[np.searchsorted(I_cumsum, self._rng.uniform(size=self.nrays))]
        # print(f"{tth.min()=}, {tth.max()=}")
        # TODO only generate extra random numbers if needed
        if self.vertical_divergence > 0:
            v_sigma = self.vertical_divergence / 2.355
            extra_vertical = self._rng.normal(scale=v_sigma, size=self.nrays)
        else:
            extra_vertical = 0

        if self.horizontal_divergence > 0:
            h_sigma = self.horizontal_divergence / 2.355
            extra_horizontal = self._rng.normal(scale=h_sigma, size=self.nrays)
        else:
            extra_horizontal = 0

        a = np.tan(np.deg2rad(tth + extra_vertical))
        b = np.tan(np.deg2rad(tth + extra_horizontal))
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = b * np.cos(phi)
        axis2[:] = a * np.sin(phi)

        scale = np.sqrt(axis1**2 + axis2**2 + 1)

        axis1[:] /= scale
        axis2[:] /= scale
