import numpy as np
import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapertures


class RectangularBeamstop(rapertures.RectangularAperture):
    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``


        .. Returned values: beamLocal
        """
        import inspect

        import xrt.backends.raycing.sources as rs

        if self.bl is not None:
            self.bl.auto_align(self, beam)
        good = beam.state > 0
        # beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        bl = self.bl if self.xyz == "auto" else self.xyz
        raycing.global_to_virgin_local(bl, beam, lo, self.center, good)
        path = -lo.y[good] / lo.b[good]
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.path[good] += path

        badIndices = np.zeros(len(beam.x), dtype=bool)
        for akind, d in zip(self.kind, self.opening, strict=True):
            if akind.startswith("l"):
                badIndices[good] = badIndices[good] | (lo.x[good] < d)
            elif akind.startswith("r"):
                badIndices[good] = badIndices[good] | (lo.x[good] > d)
            elif akind.startswith("b"):
                badIndices[good] = badIndices[good] | (lo.z[good] < d)
            elif akind.startswith("t"):
                badIndices[good] = badIndices[good] | (lo.z[good] > d)
        beam.state[~badIndices] = self.lostNum
        lo.state[:] = beam.state
        lo.y[good] = 0.0

        if hasattr(lo, "Es"):
            propPhase = np.exp(1e7j * (lo.E[good] / rapertures.CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        goodN = lo.state > 0
        try:
            self.spotLimits = [
                min(self.spotLimits[0], lo.x[goodN].min()),
                max(self.spotLimits[1], lo.x[goodN].max()),
                min(self.spotLimits[2], lo.z[goodN].min()),
                max(self.spotLimits[3], lo.z[goodN].max()),
            ]
        except ValueError:
            pass

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            raycing.append_to_flow(self.propagate, [glo, lo], inspect.currentframe())
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo], inspect.currentframe())
            return lo
