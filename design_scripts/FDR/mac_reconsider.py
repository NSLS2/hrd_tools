# %% [markdown]
# # Re-imaging MAC
#
# While a rotation arm per cyrstal may seem ideal -- it keeps
# each detector normal to the reflected beam and allows for per-
# detector alignment -- it comes at a high complexity requiring
# a stepper motor per arm.  On solution is to gang-move all of the
# arms with a single motor (as done at 11BM at APS) however this gives
# up the benefit of aligning each detector seperately.
#
# A second solution is to have the detectors on a flat plate
# that pivots about the center of the MAC.  This is a common
# design and is used at DLS and ESRF, however this relies
# on having a large active area (either a large Eiger at ESRF or
# large area scintilators at DLS and the Brazzilian light source)
# Given our choice to use many small (1.5cm x 1.5cm) area detectors
# this solution would require additional motion to ensure that the
# reflected rays landed on the detector [TODO CHECK THIS]
#
# The third solution is to
#
# ## Cases considered
#
# - **Case A**: N independent arms, each with its own rotation motor.  Each detector
#   is mounted on a rotating arm that tracks the reflected beam from its crystal.
#   The crystal-to-detector distance is fixed at Rd; the detector face is always
#   normal to the reflected beam.
#
# - **Case B**: Detectors are mounted on a rigid annular plate that rotates about the
#   sample position.  The plate is at a fixed radius D from the sample.  The
#   detector for each crystal sits at the point on the plate circle where the
#   reflected beam intersects it.  Both rd and angle of incidence vary with energy.
#
# - **Case C**: Detectors are mounted on a flat plate.  The plate centre sits at the
#   point where the central crystal's reflected beam intersects a circle of radius Rd,
#   and the plate is always perpendicular to the central crystal's beam (tracking it
#   as energy changes).  Outer crystals' beams hit the plate at a constant angle equal
#   to their angular offset from the central crystal, but rd varies with energy.

# %%
from __future__ import annotations

from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from numpy.typing import NDArray

from hrd_tools.config import CompleteConfig
from hrd_tools.xrt import CrystalProperties
import _fdr_params

# %%
_blessed: CompleteConfig = _fdr_params.complete_config()

# %% [markdown]
# ## Shared geometry setup

# %%
analyzer = CrystalProperties.create(10, t=10)
E_range: NDArray[np.float64] = np.linspace(12, 40, 128)
# Bragg angle in degrees at each energy point
angles: NDArray[np.float64] = np.array(
    [analyzer.with_energy(E).bragg_angle for E in E_range]
)

R: float = _blessed.analyzer.R  # sample to crystal, mm
Rd: float = _blessed.analyzer.Rd  # crystal to detector (Case A), mm
N: int = _blessed.analyzer.N  # number of crystals
cry_offset: float = (
    _blessed.analyzer.cry_offset
)  # angular spacing between crystals, deg

# Symmetric crystal offsets about MacTwoTheta so that both +/- wings are present
MacTwoTheta: float = 52.0  # central arm angle, deg
half_span: float = (N - 1) / 2.0 * cry_offset
crystal_offsets: NDArray[np.float64] = np.arange(N) * cry_offset - half_span
# Absolute two-theta position of each crystal arm
TwoThetas: NDArray[np.float64] = MacTwoTheta + crystal_offsets

# %% [markdown]
# ## Geometry helpers

# %%


class Line(NamedTuple):
    """A line in slope-intercept form: y = m*x + b."""

    m: float | NDArray[np.float64]
    b: float | NDArray[np.float64]


class Pt(NamedTuple):
    """A 2-D point (or pair of arrays of x, y coordinates)."""

    x: float | NDArray[np.float64]
    y: float | NDArray[np.float64]


def generate_line(theta: float | NDArray[np.float64], pt: Pt) -> Line:
    """
    Construct a line from an angle and a point on that line.

    Parameters
    ----------
    theta:
        Angle of the line from the x-axis, in degrees.
    pt:
        A point the line passes through.

    Returns
    -------
    Line
        Slope-intercept form of the line.
    """
    _theta = np.deg2rad(theta)
    m = np.tan(_theta)
    b = pt.y - m * pt.x
    return Line(m, b)


def get_intersection(a: Line, b: Line) -> NDArray[np.float64]:
    """
    Return the intersection point of two lines as a (2, ...) array [x, y].

    Parameters
    ----------
    a, b:
        Lines in slope-intercept form.  ``m`` and ``b`` may be arrays
        of shape (E,), in which case the result has shape (2, E).

    Returns
    -------
    NDArray[np.float64]
        Shape (2, ...) array where [0] is x and [1] is y.
    """
    x = (b.b - a.b) / (a.m - b.m)
    y = x * a.m + a.b
    return np.asarray([x, y])


def from_center_triangle(
    R: float,
    D: float,
    theta_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve the sample–crystal–detector triangle when the detector lies on a
    circle of radius *D* centred on the sample.

    Parameters
    ----------
    R:
        Sample-to-crystal distance, mm.
    D:
        Sample-to-detector distance (radius of detector circle), mm.
    theta_b:
        Bragg angle(s) in degrees.

    Returns
    -------
    rd:
        Crystal-to-detector distance, mm.
    beta:
        Angular offset of the detector from the crystal arm direction
        (detector angle = TwoTheta - beta), degrees.
    gamma:
        Angle of incidence of the reflected beam on the detector face
        (angle between the beam and the detector normal), degrees.
    """
    _theta_b = np.deg2rad(theta_b)
    a = 1.0
    b = 2.0 * R * np.cos(2.0 * _theta_b)
    c = R**2 - D**2
    rd = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
    beta = np.rad2deg(np.arcsin(rd * np.sin(np.pi - 2.0 * _theta_b) / D))
    gamma = np.rad2deg(np.arcsin(R * np.sin(np.pi - 2.0 * _theta_b) / D))
    return rd, beta, gamma


# %% [markdown]
# ## Per-case geometry functions

# %%


def case_A_detector_pos(
    TwoTheta: float,
    R: float,
    Rd: float,
    angles: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case A: detector arm tracks the reflected beam exactly.

    The detector sits at distance *Rd* from the crystal along the
    diffracted-beam direction ``TwoTheta - 2*theta_b``.

    Parameters
    ----------
    TwoTheta:
        Crystal arm two-theta position, degrees.
    R:
        Sample-to-crystal distance, mm.
    Rd:
        Crystal-to-detector distance, mm (fixed by construction in Case A).
    angles:
        Bragg angles in degrees, shape (E,).

    Returns
    -------
    x, y:
        Detector position arrays, shape (E,), mm.
    """
    x0 = R * np.cos(np.deg2rad(TwoTheta))
    y0 = R * np.sin(np.deg2rad(TwoTheta))
    phi = np.deg2rad(TwoTheta - 2.0 * angles)
    return x0 + Rd * np.cos(phi), y0 + Rd * np.sin(phi)


def case_A_metrics(
    TwoTheta: float,
    R: float,
    Rd: float,
    angles: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case A: crystal-to-detector distance and beam incidence angle on the detector face.

    By construction both are constant: *rd* = Rd and incidence angle = 0 deg.

    Returns
    -------
    rd:
        Crystal-to-detector distances, shape (E,), mm.
    incidence_angle:
        Angle of beam on detector face normal, degrees, shape (E,).
    """
    rd = np.full_like(angles, Rd)
    incidence_angle = np.zeros_like(angles)
    return rd, incidence_angle


def case_B_detector_pos(
    TwoTheta: float,
    D: float,
    beta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case B: detector on a rigid circle of radius *D* about the sample.

    Parameters
    ----------
    TwoTheta:
        Crystal arm two-theta, degrees.
    D:
        Sample-to-detector circle radius, mm.
    beta:
        Angular offset of the detector from the crystal arm direction,
        degrees (from :func:`from_center_triangle`), shape (E,).

    Returns
    -------
    x, y:
        Detector position arrays, shape (E,), mm.
    """
    phi = np.deg2rad(TwoTheta - beta)
    return D * np.cos(phi), D * np.sin(phi)


def case_C_plate_geometry(
    MacTwoTheta: float,
    R: float,
    Rd: float,
    cry_offset: float,
    N: int,
    angles: NDArray[np.float64],
) -> tuple[Pt, Line]:
    """
    Case C: flat plate perpendicular to the central crystal's diffracted beam.

    The rotation centre is placed at the mean crystal position (= MacTwoTheta,
    since offsets are symmetric).  The plate centre moves along the central
    crystal's diffracted-beam direction at distance *Rd*.  The plate is oriented
    perpendicular to that beam direction.

    Parameters
    ----------
    MacTwoTheta:
        Central arm two-theta (= mean of all crystal two-thetas), degrees.
    R:
        Sample-to-crystal, mm.
    Rd:
        Nominal crystal-to-plate-centre distance, mm.
    cry_offset:
        Angular spacing between adjacent crystals, degrees.
    N:
        Total number of crystals.
    angles:
        Bragg angles in degrees, shape (E,).

    Returns
    -------
    rot_center:
        Fixed rotation centre of the plate (at mean crystal position), Pt.
    plate_line:
        The flat-plate line for each energy, Line with arrays of shape (E,).
    """
    rot_center = Pt(
        R * np.cos(np.deg2rad(MacTwoTheta)),
        R * np.sin(np.deg2rad(MacTwoTheta)),
    )
    # Central crystal's diffracted-beam direction
    beam_dir = MacTwoTheta - 2.0 * angles  # degrees, shape (E,)
    # Plate centre: rot_center displaced by Rd along the central beam direction
    plate_center = Pt(
        rot_center.x + Rd * np.cos(np.deg2rad(beam_dir)),
        rot_center.y + Rd * np.sin(np.deg2rad(beam_dir)),
    )
    # Plate orientation: perpendicular to the central beam
    theta_plate = beam_dir + 90.0
    plate_line = generate_line(theta_plate, plate_center)
    return rot_center, plate_line


def case_C_detector_pos(
    TwoTheta: float,
    R: float,
    angles: NDArray[np.float64],
    plate_line: Line,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case C: intersection of crystal *TwoTheta*'s reflected ray with the flat plate.

    Parameters
    ----------
    TwoTheta:
        Crystal arm two-theta, degrees.
    R:
        Sample-to-crystal, mm.
    angles:
        Bragg angles in degrees, shape (E,).
    plate_line:
        Flat-plate line (arrays of slope/intercept, shape (E,)).

    Returns
    -------
    x, y:
        Detector position arrays, shape (E,), mm.
    """
    x0 = R * np.cos(np.deg2rad(TwoTheta))
    y0 = R * np.sin(np.deg2rad(TwoTheta))
    ray_dir = TwoTheta - 2.0 * angles  # degrees, shape (E,)
    ray = generate_line(ray_dir, Pt(x0, y0))
    pt = get_intersection(ray, plate_line)
    return pt[0], pt[1]


def case_C_metrics(
    TwoTheta: float,
    R: float,
    MacTwoTheta: float,
    angles: NDArray[np.float64],
    plate_line: Line,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case C: crystal-to-detector distance and beam incidence angle on the flat plate.

    The plate is always perpendicular to the central crystal's beam, so the
    incidence angle for crystal at *TwoTheta* equals its offset from centre
    (constant vs energy), while *rd* varies.

    Parameters
    ----------
    TwoTheta:
        Crystal arm two-theta, degrees.
    R:
        Sample-to-crystal, mm.
    MacTwoTheta:
        Central arm two-theta (plate tracks this crystal), degrees.
    angles:
        Bragg angles in degrees, shape (E,).
    plate_line:
        Flat-plate line, Line with arrays of shape (E,).

    Returns
    -------
    rd:
        Crystal-to-detector distance, mm, shape (E,).
    incidence_angle:
        Angle between reflected ray and detector face normal, degrees, shape (E,).
        Constant vs energy; equals ``TwoTheta - MacTwoTheta``.
    """
    x0 = R * np.cos(np.deg2rad(TwoTheta))
    y0 = R * np.sin(np.deg2rad(TwoTheta))
    dx, dy = case_C_detector_pos(TwoTheta, R, angles, plate_line)
    rd = np.hypot(dx - x0, dy - y0)
    incidence_angle = np.full_like(angles, TwoTheta - MacTwoTheta)
    return rd, incidence_angle


def case_D_plate_geometry(
    MacTwoTheta: float,
    R: float,
    Rd: float,
    plate_normal_angle: float | None = None,
) -> tuple[Pt, Line]:
    """
    Case D: fixed flat plate rotating about the sample position (0, 0).

    The plate is positioned at distance ``R + Rd`` from the origin along
    ``plate_normal_angle`` and oriented perpendicular to that direction.
    Unlike Case C the plate does **not** track the beam as energy changes.

    The default ``plate_normal_angle = MacTwoTheta`` points the plate normal
    along the central arm direction.  Passing the numerically optimised angle
    (see the Case D setup cell) minimises the maximum rigid-body residual
    across the energy range.

    Parameters
    ----------
    MacTwoTheta:
        Central arm two-theta, degrees.
    R:
        Sample-to-crystal, mm.
    Rd:
        Nominal central-crystal-to-plate distance, mm.
    plate_normal_angle:
        Direction the plate normal points, degrees.  Defaults to
        ``MacTwoTheta`` (plate perpendicular to the central arm).

    Returns
    -------
    rot_center:
        Fixed rotation centre of the plate: the origin (0, 0).
    plate_line:
        The fixed flat-plate line (scalar slope and intercept).
    """
    if plate_normal_angle is None:
        plate_normal_angle = MacTwoTheta
    rot_center = Pt(0.0, 0.0)
    plate_center = Pt(
        (R + Rd) * np.cos(np.deg2rad(plate_normal_angle)),
        (R + Rd) * np.sin(np.deg2rad(plate_normal_angle)),
    )
    plate_line = generate_line(plate_normal_angle + 90.0, plate_center)
    return rot_center, plate_line


def case_D_metrics(
    TwoTheta: float,
    R: float,
    MacTwoTheta: float,
    angles: NDArray[np.float64],
    plate_line: Line,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Case D: crystal-to-detector distance and beam incidence angle on the fixed plate.

    Because the plate is fixed, both rd and incidence angle vary with energy.

    Parameters
    ----------
    TwoTheta:
        Crystal arm two-theta, degrees.
    R:
        Sample-to-crystal, mm.
    MacTwoTheta:
        Central arm two-theta (plate normal direction), degrees.
    angles:
        Bragg angles in degrees, shape (E,).
    plate_line:
        Fixed flat-plate line (scalar slope and intercept).

    Returns
    -------
    rd:
        Crystal-to-detector distance, mm, shape (E,).
    incidence_angle:
        Angle between reflected ray and plate normal, degrees, shape (E,).
        Equals ``(TwoTheta - MacTwoTheta) - 2 * angles``.
    """
    x0 = R * np.cos(np.deg2rad(TwoTheta))
    y0 = R * np.sin(np.deg2rad(TwoTheta))
    dx, dy = case_C_detector_pos(TwoTheta, R, angles, plate_line)
    rd = np.hypot(dx - x0, dy - y0)
    # Plate normal points along MacTwoTheta; ray direction is TwoTheta - 2*angles
    incidence_angle = (TwoTheta - MacTwoTheta) - 2.0 * angles
    return rd, incidence_angle


# %% [markdown]
# ## Shared plotting helpers

# %%


def crystal_colors(n: int) -> list:
    """
    Return *n* colours sampled from the magma colormap, avoiding the very dark
    ends so all traces are legible on a white background.

    Parameters
    ----------
    n:
        Number of colours needed (one per crystal).

    Returns
    -------
    list
        List of RGBA tuples of length *n*.
    """
    cmap = matplotlib.colormaps["magma"]
    return [cmap(v) for v in np.linspace(0.15, 0.85, n)]


def plot_realspace(
    ax: Axes,
    TwoThetas: NDArray[np.float64],
    R: float,
    E_range: NDArray[np.float64],
    det_pos_fn,
    *,
    det_circle_r: float | None = None,
    extra_circles: list[tuple[Pt, float, str]] | None = None,
):
    """
    Plot the real-space layout of crystals and their detector positions vs energy.

    Crystal markers are coloured by crystal index (magma).  Detector scatter
    points are coloured by energy (viridis).

    Parameters
    ----------
    ax:
        Axes to draw on.
    TwoThetas:
        Crystal arm angles, degrees.
    R:
        Sample-to-crystal distance, mm.
    E_range:
        Energy values for colouring scatter points, keV.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` returning detector positions as
        arrays of length ``len(E_range)``.
    det_circle_r:
        If given, draw a reference circle of this radius (mm) about the origin.
    extra_circles:
        List of ``(center_Pt, radius, linestyle)`` for additional circles.

    Returns
    -------
    sc:
        The scatter artist (for attaching an energy colorbar).
    """
    norm = Normalize(E_range.min(), E_range.max())
    cmap = matplotlib.colormaps["viridis"]

    colors = crystal_colors(len(TwoThetas))
    sc = None
    for i, (TwoTheta, color) in enumerate(zip(TwoThetas, colors)):
        x0 = R * np.cos(np.deg2rad(TwoTheta))
        y0 = R * np.sin(np.deg2rad(TwoTheta))
        ax.plot(x0, y0, marker="o", color=color, ms=3, zorder=3)
        dx, dy = det_pos_fn(TwoTheta)
        sc = ax.scatter(dx, dy, c=E_range, norm=norm, cmap=cmap, s=4, zorder=2)

    if det_circle_r is not None:
        ax.add_patch(
            Circle(
                (0, 0),
                det_circle_r,
                facecolor="none",
                lw=1,
                linestyle=":",
                edgecolor="gray",
                alpha=0.5,
            )
        )
    if extra_circles:
        for center, radius, ls in extra_circles:
            ax.add_patch(
                Circle(
                    (center.x, center.y),
                    radius,
                    facecolor="none",
                    lw=1,
                    linestyle=ls,
                    edgecolor="gray",
                    alpha=0.5,
                )
            )

    ax.set_aspect("equal")
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    return sc


def plot_rd_and_angle(
    ax_rd: Axes,
    ax_angle: Axes,
    TwoThetas: NDArray[np.float64],
    E_range: NDArray[np.float64],
    metrics_fn,
) -> None:
    """
    Plot crystal-to-detector distance and beam incidence angle vs energy.

    One line per crystal, coloured by crystal index (magma).

    Parameters
    ----------
    ax_rd:
        Axes for the crystal-to-detector distance.
    ax_angle:
        Axes for the beam incidence angle.
    TwoThetas:
        Crystal arm angles, degrees.
    E_range:
        Energy array, keV.
    metrics_fn:
        Callable ``f(TwoTheta) -> (rd, incidence_angle)`` returning arrays of
        length ``len(E_range)``.
    """
    colors = crystal_colors(len(TwoThetas))
    for i, (TwoTheta, color) in enumerate(zip(TwoThetas, colors)):
        rd, angle = metrics_fn(TwoTheta)
        label = f"{TwoTheta:.0f}°"
        ax_rd.plot(E_range, rd, color=color, label=label)
        ax_angle.plot(E_range, angle, color=color, label=label)

    ax_rd.set_ylabel("Crystal–detector\ndistance (mm)")
    ax_angle.set_ylabel("Incidence\nangle (deg)")
    ax_angle.set_xlabel("Energy (keV)")


def plot_adjacent_spacing(
    ax: Axes,
    TwoThetas: NDArray[np.float64],
    MacTwoTheta: float,
    E_range: NDArray[np.float64],
    det_pos_fn,
) -> None:
    """
    Distance between adjacent detector hit positions vs energy.

    One line per adjacent crystal pair, coloured by index (magma).  If all
    lines are flat the detector array moves as a single rigid body; differing
    slopes mean independent per-detector motion is needed.

    Parameters
    ----------
    ax:
        Axes to draw on.
    TwoThetas:
        Crystal arm angles in degrees, ordered from lowest to highest.
    MacTwoTheta:
        Central arm angle (offset = 0), degrees.  Used for pair labels.
    E_range:
        Energy array, keV.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` returning detector positions,
        shape (E,) each.
    """
    pair_colors = crystal_colors(len(TwoThetas) - 1)

    # Compute change in spacing for each adjacent pair
    delta_spacings = []
    for i in range(len(TwoThetas) - 1):
        tt0, tt1 = TwoThetas[i], TwoThetas[i + 1]
        x0, y0 = det_pos_fn(tt0)
        x1, y1 = det_pos_fn(tt1)
        spacing = np.hypot(x1 - x0, y1 - y0)
        delta_spacings.append(spacing - spacing[0])

    # Cumulative sum: detector k must move by sum of delta_spacings[0..k-1]
    # relative to detector 0 being held fixed.  Plot one line per crystal
    # (skip crystal 0 which is the fixed reference).
    cumsum = np.zeros_like(delta_spacings[0])
    for i, (delta, color) in enumerate(zip(delta_spacings, pair_colors)):
        cumsum = cumsum + delta
        label = f"{TwoThetas[i + 1] - MacTwoTheta:+.0f}°"
        ax.plot(E_range, cumsum, color=color, label=label)

    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Required in-plane\nmotion (mm)")


def _compute_rigid_transform(
    TwoThetas: NDArray[np.float64],
    E_range: NDArray[np.float64],
    det_pos_fn,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Compute the best-fit rigid transform (rotation + translation) at each energy.

    At each energy the Kabsch algorithm finds the rotation ``R`` and translation
    ``t`` that minimise the sum of squared distances between the lowest-energy
    detector configuration and the current one.  Both configurations are
    centred on the E[0] detector array centroid before the SVD, so that ``t``
    is the residual translation *after* the best rotation has been removed — a
    small number whenever the motion is close to a pure rotation, regardless of
    where the array sits in the lab frame.

    Parameters
    ----------
    TwoThetas:
        Crystal arm angles, degrees.
    E_range:
        Energy array, keV.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` returning detector positions,
        shape (E,) each.

    Returns
    -------
    theta : NDArray, shape (E,)
        Rotation angle in degrees (positive = anticlockwise).
    tx, ty : NDArray, shape (E,)
        Residual translation after the best rotation, in the frame centred on
        the E[0] array centroid, mm.  Near-zero when the motion is a pure
        rotation about any fixed point.
    residuals : NDArray, shape (N, E)
        Per-detector residual magnitude after applying the rigid transform, mm.
    """
    positions = np.array([np.stack(det_pos_fn(tt)) for tt in TwoThetas])  # (N, 2, E)
    p0 = positions[:, :, 0]  # (N, 2) reference
    p0_mean = p0.mean(axis=0)  # (2,) — centroid of reference config
    p0_centered = p0 - p0_mean  # (N, 2)

    N, _, E = positions.shape
    theta = np.zeros(E)
    tx = np.zeros(E)
    ty = np.zeros(E)
    residuals = np.zeros((N, E))

    for ei in range(E):
        p = positions[:, :, ei]  # (N, 2)
        p_mean = p.mean(axis=0)
        p_centered = p - p_mean

        H = p0_centered.T @ p_centered  # (2, 2) cross-covariance
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Guard against reflections (det = -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        theta[ei] = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))

        # Residual translation: how far does the centroid move after the best
        # rotation has been applied?  Both centroids are expressed relative to
        # p0_mean so that the large lab-frame offset cancels.
        # t = (p_mean - p0_mean) - (R - I) @ p0_mean
        #   = centroid_drift - rotation_induced_centroid_swing
        # This is small when the motion is a pure rotation about any fixed point.
        t = (p_mean - p0_mean) - (R - np.eye(2)) @ p0_mean
        tx[ei] = t[0]
        ty[ei] = t[1]

        p0_aligned = (R @ p0_centered.T).T + p_mean
        residuals[:, ei] = np.hypot(*(p - p0_aligned).T)

    return theta, tx, ty, residuals


def plot_rigid_residual(
    ax: Axes,
    TwoThetas: NDArray[np.float64],
    MacTwoTheta: float,
    E_range: NDArray[np.float64],
    det_pos_fn,
) -> None:
    """
    Residual detector motion after removing the best-fit rigid shift+rotation.

    If all residuals are near zero a single rigid-body motion suffices.
    Non-zero residuals indicate detectors that must move independently.

    Parameters
    ----------
    ax:
        Axes to draw on.
    TwoThetas:
        Crystal arm angles, degrees.
    MacTwoTheta:
        Central arm angle (offset = 0), degrees.  Used for labels.
    E_range:
        Energy array, keV.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` returning detector positions,
        shape (E,) each.
    """
    _, _, _, residuals = _compute_rigid_transform(TwoThetas, E_range, det_pos_fn)
    colors = crystal_colors(len(TwoThetas))

    for i, (TwoTheta, color) in enumerate(zip(TwoThetas, colors)):
        ax.plot(
            E_range, residuals[i], color=color, label=f"{TwoTheta - MacTwoTheta:+.0f}°"
        )

    # Shade the ±7.5 mm band (half of the 15 mm detector active height).
    # Residuals that stay within this band are captured by a single detector
    # without any independent motion.
    ax.axhspan(0, 7.5, color="green", alpha=0.08, zorder=0)
    ax.axhline(7.5, color="green", lw=0.8, linestyle="--", label="7.5 mm (½ detector)")
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Rigid-body\nresidual (mm)")


def plot_rigid_transform(
    ax_theta: Axes,
    ax_cor: Axes,
    E_range: NDArray[np.float64],
    det_pos_fn,
    TwoThetas: NDArray[np.float64],
) -> None:
    """
    Plot the ideal rigid-body transform parameters vs energy.

    Shows the rotation angle and the drift of the implied centre of rotation
    (CoR) from its lowest-energy position.  A fixed CoR means the motion is
    a pure rotation about a single point and only one rotation motor is needed.
    A drifting CoR means an additional translation motor is required.

    For cases with negligible rotation (Case D) the CoR is undefined; the
    axes are left blank.

    Parameters
    ----------
    ax_theta:
        Axes for the rotation angle.
    ax_cor:
        Axes for the CoR drift magnitude.
    E_range:
        Energy array, keV.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` returning detector positions,
        shape (E,) each.
    TwoThetas:
        Crystal arm angles, degrees.  Used only to call ``det_pos_fn``.
    """
    theta, tx, ty, _ = _compute_rigid_transform(TwoThetas, E_range, det_pos_fn)

    ax_theta.plot(E_range, theta)
    ax_theta.axhline(0, color="k", lw=0.6, linestyle="--")
    ax_theta.set_ylabel("Rotation\nangle (deg)")
    ax_theta.annotate(
        f"max = {theta.max():.3g}°",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(4, -4),
        textcoords="offset points",
        fontsize="x-small",
        va="top",
        ha="left",
    )

    # Compute the implied centre of rotation at each energy from the SVD.
    # CoR is the fixed point of the transform: (I - R) @ cor = t_svd.
    # Only defined when |theta| is large enough that (I - R) is invertible.
    positions = np.array([np.stack(det_pos_fn(tt)) for tt in TwoThetas])
    p0 = positions[:, :, 0]
    p0_mean = p0.mean(axis=0)
    p0_centered = p0 - p0_mean

    cor_drift = np.full(len(E_range), np.nan)
    cor0 = None
    for ei in range(len(E_range)):
        if abs(theta[ei]) < 0.01:  # skip near-zero rotation — CoR undefined
            continue
        p = positions[:, :, ei]
        p_mean = p.mean(axis=0)
        p_centered = p - p_mean
        H = p0_centered.T @ p_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t_svd = p_mean - R @ p0_mean
        IminR = np.eye(2) - R
        if abs(np.linalg.det(IminR)) < 1e-10:
            continue
        cor = np.linalg.solve(IminR, t_svd)
        if cor0 is None:
            cor0 = cor
        cor_drift[ei] = np.hypot(*(cor - cor0))

    if not np.all(np.isnan(cor_drift)):
        ax_cor.plot(E_range, cor_drift)
        ax_cor.axhline(0, color="k", lw=0.6, linestyle="--")
        valid = cor_drift[~np.isnan(cor_drift)]
        ax_cor.annotate(
            f"max = {np.nanmax(cor_drift):.3g} mm",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(4, -4),
            textcoords="offset points",
            fontsize="x-small",
            va="top",
            ha="left",
        )
    else:
        ax_cor.text(
            0.5,
            0.5,
            "CoR undefined\n(no rotation)",
            transform=ax_cor.transAxes,
            ha="center",
            va="center",
            fontsize="small",
            color="gray",
        )

    ax_cor.set_ylabel("CoR drift (mm)")
    ax_cor.set_xlabel("Energy (keV)")


def _crystal_index_colorbar(
    fig, ax: Axes, TwoThetas: NDArray[np.float64], MacTwoTheta: float
) -> None:
    """
    Add a colorbar showing crystal index → magma colour alongside *ax*.

    Tick labels show the arm offset in degrees.

    Parameters
    ----------
    fig:
        Parent figure.
    ax:
        Axes to attach the colorbar to.
    TwoThetas:
        Crystal arm two-theta positions, degrees.
    MacTwoTheta:
        Central arm two-theta, degrees.
    """
    N = len(TwoThetas)
    cmap_m = matplotlib.colormaps["magma"]
    norm_m = Normalize(0, N - 1)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap_m, norm=norm_m)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, location="right", fraction=0.046, pad=0.04)
    cb.set_label("Crystal index")
    cb.set_ticks(np.arange(N))
    cb.set_ticklabels([f"{tt - MacTwoTheta:+.0f}°" for tt in TwoThetas])


def plot_case(
    case_label: str,
    det_pos_fn,
    metrics_fn,
    TwoThetas: NDArray[np.float64],
    MacTwoTheta: float,
    R: float,
    E_range: NDArray[np.float64],
    *,
    realspace_kw: dict | None = None,
    extra_artists_fn=None,
):
    """
    Build a single combined figure for one detector case.

    Layout (subplot_mosaic)::

        [ realspace | rd     ]
        [ realspace | angle  ]
        [ realspace | adj    ]

    Parameters
    ----------
    case_label:
        Short label used in the suptitle, e.g. ``"A — independent arms"``.
    det_pos_fn:
        Callable ``f(TwoTheta) -> (x, y)`` for detector positions.
    metrics_fn:
        Callable ``f(TwoTheta) -> (rd, incidence_angle)``.
    TwoThetas:
        Crystal arm angles, degrees.
    MacTwoTheta:
        Central arm angle, degrees.
    R:
        Sample-to-crystal distance, mm.
    E_range:
        Energy array, keV.
    realspace_kw:
        Extra keyword arguments forwarded to :func:`plot_realspace`
        (e.g. ``det_circle_r``, ``extra_circles``).
    extra_artists_fn:
        Optional callable ``f(ax_rs)`` called after :func:`plot_realspace`
        to add case-specific annotations (plate guide lines, rotation centre
        marker, etc.).

    Returns
    -------
    fig, axes_dict
        The figure and the mosaic axes dict with keys
        ``"rs"``, ``"rd"``, ``"angle"``, ``"rigid"``,
        ``"theta"``, ``"t"``.
    """
    fig, axd = plt.subplot_mosaic(
        [
            ["rs", "rd", "rigid"],
            ["rs", "angle", "theta"],
            ["rs", "angle", "cor"],
        ],
        figsize=(16, 8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1, 1]},
    )
    ax_rs = axd["rs"]
    ax_rd = axd["rd"]
    ax_angle = axd["angle"]
    ax_rigid = axd["rigid"]
    ax_theta = axd["theta"]
    ax_cor = axd["cor"]

    # --- real-space panel ---
    sc = plot_realspace(
        ax_rs, TwoThetas, R, E_range, det_pos_fn, **(realspace_kw or {})
    )
    if extra_artists_fn is not None:
        extra_artists_fn(ax_rs)

    # Energy colorbar on the realspace panel
    cb_e = fig.colorbar(sc, ax=ax_rs, location="top", fraction=0.046, pad=0.04)
    cb_e.set_label("Energy (keV)")

    # Crystal-index colorbar alongside the realspace panel
    _crystal_index_colorbar(fig, ax_rs, TwoThetas, MacTwoTheta)

    # --- rd / incidence angle panel (shared x, stacked) ---
    plot_rd_and_angle(ax_rd, ax_angle, TwoThetas, E_range, metrics_fn)
    ax_rd.sharex(ax_angle)
    ax_rd.tick_params(labelbottom=False)

    # --- rigid-body residual panel ---
    plot_rigid_residual(ax_rigid, TwoThetas, MacTwoTheta, E_range, det_pos_fn)
    ax_rigid.sharex(ax_angle)
    ax_rigid.tick_params(labelbottom=False)

    # --- rotation angle and CoR drift panels ---
    plot_rigid_transform(ax_theta, ax_cor, E_range, det_pos_fn, TwoThetas)
    ax_theta.sharex(ax_angle)
    ax_theta.tick_params(labelbottom=False)
    ax_cor.sharex(ax_angle)

    fig.suptitle(f"Case {case_label}", fontsize=12)
    return fig, axd


# %% [markdown]
# ## Case A — N independent rotating arms

# %%


def _case_A_det(TwoTheta: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_A_detector_pos(TwoTheta, R, Rd, angles)


def _case_A_metrics(
    TwoTheta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_A_metrics(TwoTheta, R, Rd, angles)


# %%
fig_A, axd_A = plot_case(
    "A — independent rotating arms",
    _case_A_det,
    _case_A_metrics,
    TwoThetas,
    MacTwoTheta,
    R,
    E_range,
    realspace_kw={
        "extra_circles": [
            (Pt(R * np.cos(np.deg2rad(tt)), R * np.sin(np.deg2rad(tt))), Rd, "--")
            for tt in TwoThetas
        ]
    },
)
ax_A_rs = axd_A["rs"]

# %% [markdown]
# ## Case B — rigid annular plate rotating about the sample

# %%
D_B: float = Rd + R  # sample-to-detector circle radius, mm
rd_B, beta_B, gamma_B = from_center_triangle(R, D_B, angles)


def _case_B_det(TwoTheta: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_B_detector_pos(TwoTheta, D_B, beta_B)


def _case_B_metrics(
    TwoTheta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # rd and gamma are the same for every crystal in Case B
    # (all crystals are at the same distance R from the sample, so the
    # triangle is identical for each; only the arm angle differs)
    return rd_B, gamma_B


# %%
fig_B, axd_B = plot_case(
    "B — rigid annular plate",
    _case_B_det,
    _case_B_metrics,
    TwoThetas,
    MacTwoTheta,
    R,
    E_range,
    realspace_kw={"det_circle_r": D_B},
)
ax_B_rs = axd_B["rs"]

# %% [markdown]
# ## Case C — flat plate

# %%
rot_center_C, plate_line_C = case_C_plate_geometry(
    MacTwoTheta, R, Rd, cry_offset, N, angles
)


def _case_C_det(TwoTheta: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_C_detector_pos(TwoTheta, R, angles, plate_line_C)


def _case_C_metrics(
    TwoTheta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_C_metrics(TwoTheta, R, MacTwoTheta, angles, plate_line_C)


# %%
def _case_C_extra(ax_rs: Axes) -> None:
    ax_rs.plot(
        rot_center_C.x,
        rot_center_C.y,
        marker="+",
        color="k",
        ms=10,
        zorder=4,
        label="Plate rotation centre",
    )
    for _ei, _ls, _lbl in [
        (0, "--", f"Plate at {E_range[0]:.0f} keV"),
        (-1, ":", f"Plate at {E_range[-1]:.0f} keV"),
    ]:
        _pc_x = rot_center_C.x + Rd * np.cos(
            np.deg2rad(MacTwoTheta - 2.0 * angles[_ei])
        )
        _pc_y = rot_center_C.y + Rd * np.sin(
            np.deg2rad(MacTwoTheta - 2.0 * angles[_ei])
        )
        ax_rs.axline(
            (_pc_x, _pc_y),
            slope=plate_line_C.m[_ei],
            linestyle=_ls,
            color="gray",
            lw=1.5,
            zorder=1,
            label=_lbl,
        )
    ax_rs.legend(loc="lower right", fontsize="small")


fig_C, axd_C = plot_case(
    "C — flat plate (rotating about crystal)",
    _case_C_det,
    _case_C_metrics,
    TwoThetas,
    MacTwoTheta,
    R,
    E_range,
    realspace_kw={"extra_circles": [(rot_center_C, Rd, ":")]},
    extra_artists_fn=_case_C_extra,
)
ax_C_rs = axd_C["rs"]

# %% [markdown]
# ## Case D — fixed flat plate, rotating about the sample (0, 0)

# %%
# Find the plate normal angle that minimises the maximum rigid-body residual.
# The beam direction sweeps from MacTwoTheta - 2*angles[0] (low energy)
# to MacTwoTheta - 2*angles[-1] (high energy).  We search a range around
# MacTwoTheta to find the optimum.
_pn_sweep = np.linspace(
    MacTwoTheta - 2 * angles[-1] - 5,  # a little beyond the high-energy beam
    MacTwoTheta + 5,  # a little beyond the nominal arm angle
    300,
)
_pn_max_res = []
for _pn in _pn_sweep:
    _, _pl = case_D_plate_geometry(MacTwoTheta, R, Rd, _pn)

    def _tmp_det(tt, _plate=_pl):
        return case_C_detector_pos(tt, R, angles, _plate)

    _, _, _, _res = _compute_rigid_transform(TwoThetas, E_range, _tmp_det)
    _pn_max_res.append(_res.max())

_pn_max_res = np.array(_pn_max_res)
_optimal_pn: float = float(_pn_sweep[np.argmin(_pn_max_res)])
print(
    f"Optimal plate normal angle: {_optimal_pn:.2f} deg "
    f"(MacTwoTheta {MacTwoTheta:+.2f} deg offset = {_optimal_pn - MacTwoTheta:.2f} deg)"
)
print(
    f"  Max residual at MacTwoTheta:  {_pn_max_res[np.argmin(np.abs(_pn_sweep - MacTwoTheta))]:.3f} mm"
)
print(f"  Max residual at optimum:      {_pn_max_res.min():.3f} mm")

# %%
# Plot residual vs plate normal angle to visualise the optimisation
fig_D_sweep, ax_D_sweep = plt.subplots(layout="constrained")
ax_D_sweep.plot(_pn_sweep, _pn_max_res)
ax_D_sweep.axvline(
    MacTwoTheta, color="gray", linestyle="--", label=f"MacTwoTheta = {MacTwoTheta:.1f}°"
)
ax_D_sweep.axvline(
    _optimal_pn, color="C1", linestyle="--", label=f"Optimum = {_optimal_pn:.1f}°"
)
ax_D_sweep.axvline(
    MacTwoTheta - 2 * angles[-1],
    color="C2",
    linestyle=":",
    label=f"High-E beam = {MacTwoTheta - 2 * angles[-1]:.1f}°",
)
ax_D_sweep.axvline(
    MacTwoTheta - 2 * angles[0],
    color="C3",
    linestyle=":",
    label=f"Low-E beam = {MacTwoTheta - 2 * angles[0]:.1f}°",
)
ax_D_sweep.set_xlabel("Plate normal angle (deg)")
ax_D_sweep.set_ylabel("Max rigid-body residual (mm)")
ax_D_sweep.set_title("Case D — optimising fixed plate orientation")
ax_D_sweep.legend(fontsize="small")

# %%
rot_center_D, plate_line_D = case_D_plate_geometry(MacTwoTheta, R, Rd, _optimal_pn)


def _case_D_det(TwoTheta: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_C_detector_pos(TwoTheta, R, angles, plate_line_D)


def _case_D_metrics(
    TwoTheta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return case_D_metrics(TwoTheta, R, MacTwoTheta, angles, plate_line_D)


# %%
def _case_D_extra(ax_rs: Axes) -> None:
    ax_rs.plot(
        rot_center_D.x,
        rot_center_D.y,
        marker="+",
        color="k",
        ms=10,
        zorder=4,
        label="Plate rotation centre",
    )
    # plate_line_D is a fixed line (scalar m, b)
    _pc_x = (R + Rd) * np.cos(np.deg2rad(MacTwoTheta))
    _pc_y = (R + Rd) * np.sin(np.deg2rad(MacTwoTheta))
    ax_rs.axline(
        (_pc_x, _pc_y),
        slope=float(plate_line_D.m),
        linestyle="-",
        color="gray",
        lw=1.5,
        zorder=1,
        label="Plate",
    )
    ax_rs.legend(loc="lower right", fontsize="small")


fig_D, axd_D = plot_case(
    "D — fixed flat plate (rotating about sample)",
    _case_D_det,
    _case_D_metrics,
    TwoThetas,
    MacTwoTheta,
    R,
    E_range,
    realspace_kw={"extra_circles": [(rot_center_D, R + Rd, ":")]},
    extra_artists_fn=_case_D_extra,
)
ax_D_rs = axd_D["rs"]

# %%
# Unify the x/y limits across all four real-space plots so they are directly comparable.
_all_xs = []
_all_ys = []
for _det_fn in (_case_A_det, _case_B_det, _case_C_det, _case_D_det):
    for _tt in TwoThetas:
        _x, _y = _det_fn(_tt)
        _all_xs.append(_x)
        _all_ys.append(_y)
    # also include crystal positions
    _all_xs.append(R * np.cos(np.deg2rad(TwoThetas)))
    _all_ys.append(R * np.sin(np.deg2rad(TwoThetas)))

_all_xs_flat = np.concatenate(_all_xs)
_all_ys_flat = np.concatenate(_all_ys)
_pad = 20  # mm padding around the data extent
_xlim = (_all_xs_flat.min() - _pad, _all_xs_flat.max() + _pad)
_ylim = (_all_ys_flat.min() - _pad, _all_ys_flat.max() + _pad)

for _ax_rs in (ax_A_rs, ax_B_rs, ax_C_rs, ax_D_rs):
    _ax_rs.set_xlim(_xlim)
    _ax_rs.set_ylim(_ylim)
    _ax_rs.set_aspect("equal")

# %%
# Unify y-limits for each diagnostic panel type across all four cases.
for _panel in ("rd", "angle", "rigid", "theta", "cor"):
    _axes = [axd_A[_panel], axd_B[_panel], axd_C[_panel], axd_D[_panel]]
    _ymin = min(ax.get_ylim()[0] for ax in _axes)
    _ymax = max(ax.get_ylim()[1] for ax in _axes)
    for _ax in _axes:
        _ax.set_ylim(_ymin, _ymax)

# %%
# Save all four case figures to disk.
from pathlib import Path

_figdir = Path(__file__).parent / "figures"
_figdir.mkdir(exist_ok=True)

for _case_lbl, _fig in [("A", fig_A), ("B", fig_B), ("C", fig_C), ("D", fig_D)]:
    _out = _figdir / f"mac_case_{_case_lbl}.pdf"
    _fig.savefig(_out, dpi=150)
    print(f"Saved {_out}")

# %%
