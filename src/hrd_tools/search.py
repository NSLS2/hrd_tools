from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mpcolours
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation


@dataclass
class Sample:
    """A sample."""

    name: str
    metadata: dict[str, Any]


@dataclass
class LocRange:
    """The extent of the sample in motor space."""

    center: float
    width: float


@dataclass
class SampleLocation:
    """Sample metadata and location."""

    sample: Sample
    x: LocRange
    y: LocRange


@dataclass
class BboxView:
    """Merged box and annotation 'artist'."""

    rect: Rectangle
    label: Annotation


@dataclass
class SampleViz:
    """Helper to show found sample locations."""

    location: SampleLocation
    artists: BboxView | None = None

    def show_box(self, ax, callback=None):
        """
        Show the bounding box where the sample was found.

        Parameters
        ----------
        ax : Axes
            The axes to plot to.
        callback : Sample
           Callback to be passed the Sample when clicked.

        """
        xy = (
            self.location.x.center - self.location.x.width / 2,
            self.location.y.center - self.location.y.width / 2,
        )
        self.artists = BboxView(
            Rectangle(
                xy,
                self.location.x.width,
                self.location.y.width,
                facecolor=mpcolours.to_rgba("r", 0.5),
                edgecolor="k",
            ),
            Annotation(
                self.location.sample.name,
                xy,
                xytext=(1, -1),
                va="top",
                textcoords="offset points",
                picker=True,
            ),
        )
        ax.add_patch(self.artists.rect)
        ax.add_artist(self.artists.label)

        def on_pick(event):
            if event.artist is self.artists.label:
                callback(self.location.sample)

        if callback:
            ax.figure.canvas.mpl_connect("pick_event", on_pick)
