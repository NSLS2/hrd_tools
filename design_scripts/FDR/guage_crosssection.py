# %%
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials as rmats
from matplotlib.collections import PolyCollection

# %%
plt.switch_backend("qtagg")
# %%
E = 30_000
crystal = rmats.CrystalSi(t=1)

R = 910  # in mm

# in rad
darwin_rad = crystal.get_Darwin_width(E)


# %%
class Line(NamedTuple):
    """Represents a line in slope-intercept form: y = slope * x + intercept"""

    slope: float
    intercept: float


class GaugeWedge(NamedTuple):
    """Represents the three lines defining a gauge volume wedge"""

    bottom: Line
    center: Line
    top: Line


# Type alias for trapezoid corners
Point2D = tuple[float, float]
Trapezoid = tuple[Point2D, Point2D, Point2D, Point2D]


def line_from_point_angle(x: float, y: float, theta: float) -> Line:
    """
    Calculate the slope and intercept of a line passing through (x, y) at angle theta.

    Parameters
    ----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
    theta : float
        Angle with respect to the x-axis in radians

    Returns
    -------
    Line
        Named tuple containing slope and intercept (y = slope * x + intercept)

    Notes
    -----
    For vertical lines (theta = π/2), slope will be inf and intercept will be -inf.
    """
    slope = np.tan(theta)
    intercept = y - slope * x
    return Line(slope, intercept)


def crystal_gague_shape(
    R: float,
    theta: float,
    delta_theta: float = np.rad2deg(darwin_rad),
    offset: float = 0.0,
) -> GaugeWedge:
    """
    Calculate the edges an center of Gauge wedge.

    Parameters
    ----------
    R : float
        Radial distance from origin
    theta : float
        Angle in degrees
    delta_theta : float
        Angular offset in degrees
    offset : float
        Spatial offset perpendicular to the centerline direction (same units as R).
        The edge lines are offset by +offset and -offset, while the center line remains at 0.

    Returns
    -------
    GaugeWedge
        Named tuple containing the three lines (bottom, center, top) defining the gauge wedge
    """
    # Convert degrees to radians
    theta_rad = np.deg2rad(theta)
    delta_theta_rad = np.deg2rad(delta_theta)

    # Convert polar to Cartesian coordinates
    x = R * np.cos(theta_rad)
    y = R * np.sin(theta_rad)

    # Calculate perpendicular offset components (rotate by 90 degrees)
    dx_perp = np.cos(theta_rad + np.pi / 2)
    dy_perp = np.sin(theta_rad + np.pi / 2)

    # Calculate slope and intercept for each of the three angles with appropriate offsets
    return GaugeWedge(
        bottom=line_from_point_angle(
            x - offset * dx_perp, y - offset * dy_perp, theta_rad - delta_theta_rad
        ),
        center=line_from_point_angle(x, y, theta_rad),
        top=line_from_point_angle(
            x + offset * dx_perp, y + offset * dy_perp, theta_rad + delta_theta_rad
        ),
    )


def gague_corners(
    line1: Line, line2: Line, y_lower: float, y_upper: float
) -> Trapezoid:
    """
    Calculate the four corner points where two lines intersect horizontal bounds.

    Parameters
    ----------
    line1 : Line
        First line (slope, intercept)
    line2 : Line
        Second line (slope, intercept)
    y_lower : float
        Lower horizontal bound
    y_upper : float
        Upper horizontal bound

    Returns
    -------
    Trapezoid
        Four (x, y) points in clockwise order starting from top-left:
        (top-left, top-right, bottom-right, bottom-left)
    """
    m1, b1 = line1
    m2, b2 = line2

    # Calculate x-coordinates where each line intersects the bounds
    # For line: y = mx + b, solve for x: x = (y - b) / m
    x1_upper = (y_upper - b1) / m1
    x2_upper = (y_upper - b2) / m2
    x1_lower = (y_lower - b1) / m1
    x2_lower = (y_lower - b2) / m2

    # Determine which line is on the left at upper bound
    if x1_upper < x2_upper:
        # line1 is on the left
        top_left = (x1_upper, y_upper)
        top_right = (x2_upper, y_upper)
        bottom_right = (x2_lower, y_lower)
        bottom_left = (x1_lower, y_lower)
    else:
        # line2 is on the left
        top_left = (x2_upper, y_upper)
        top_right = (x1_upper, y_upper)
        bottom_right = (x1_lower, y_lower)
        bottom_left = (x2_lower, y_lower)

    return (top_left, top_right, bottom_right, bottom_left)


# %%
# Figure 1
fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
x_vals = np.array([-5, 1000])
angles = [5, 45, 85]
colors = mpl.colormaps["tab10"](range(3))  # Get 3 distinct colors
# Define styles for the three lines: [lower edge, center, upper edge]
line_styles = [
    {},  # Lower edge: solid, full alpha
    {"linestyle": "--", "alpha": 0.5},  # Center: dashed, half alpha
    {},  # Upper edge: solid, full alpha
]
for angle, color in zip(angles, colors, strict=True):
    wedge = crystal_gague_shape(R, angle, offset=75.0 / 1000 * 30 / 2)
    for idx, (ln, style) in enumerate(zip(wedge, line_styles, strict=True)):
        slope, intercept = ln
        y_vals = slope * x_vals + intercept
        # Only add label for the first line of each angle group
        label = f"$2\\Theta$={angle:.2f}°" if idx == 0 else None
        ax.plot(x_vals, y_vals, color=color, label=label, **style)
ax.axhspan(-0.5, 0.5, color="gray", alpha=0.5, label="Incoming beam")
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-2.0, 2.0)
ax.legend()
plt.show()

# %%
# Figure 2
fig, ax = plt.subplots(layout="constrained")
angles = np.linspace(1, 90, 128)
# Collect all polygons
polygons = []
for angle in angles:
    wedge = crystal_gague_shape(R, angle, offset=75.0 / 1000 * 40 / 2)
    corners = gague_corners(wedge.bottom, wedge.top, -0.5, 0.5)
    polygons.append(corners)

# Create PolyCollection with colormap
poly_collection = PolyCollection(
    polygons,
    array=np.array(angles),  # Color based on angle values
    cmap="viridis",
    facecolors="none",
    linewidths=2,
    alpha=1.0,
)
ax.add_collection(poly_collection)

# Add colorbar horizontally below the axes
cbar = fig.colorbar(
    poly_collection,
    ax=ax,
    label="$2\\Theta$ (degrees)",
    orientation="horizontal",
    location="bottom",
)

ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-0.75, 0.75)
# ax.set_aspect("equal")
ax.set_xlabel("z (mm)")
ax.set_ylabel("y (mm)")
plt.show()
# %%
