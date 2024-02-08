"""
This module contains definitions of terrains and obstacles.
"""

from shapely import Polygon, union_all
from shapely.plotting import plot_polygon, plot_line, plot_points
from shapely.geometry.multipolygon import MultiPolygon

# create two polygons
polygon_1 = Polygon([
    (9, 6),
    (11, 6),
    (11, 5.5),
    (10, 5),
    (12, 5),
    (14, 7),
    (9, 7)
])

polygon_2 = Polygon([
    (10, 8),
    (12, 8),
    (12, 11),
    (6, 11)
])

polygon_3 = Polygon([
    (8, 2),
    (9, 2),
    (9, 4),
    (8, 4)
])

# terrains = [polygon_1, polygon_2, polygon_3]
terrains = [polygon_3]


def get_terrains_union(buffer=0):
    """Returns union of all terrains as list of polygons. Buffer possible.
    """
    buffered = [terrain.buffer(buffer, quad_segs=1) for terrain in terrains]
    union = union_all(buffered)
    if isinstance(union, MultiPolygon):
        return list(union.geoms)
    else:
        return [union]


# x = get_terrains_union(0.98 / 2)
# isinstance(x, MultiPolygon)
# type(x)
# list(x.geoms)