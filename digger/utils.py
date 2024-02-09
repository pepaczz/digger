"""
Contains mostly movement related functions.
Uses mainly shapely and pyvisgraph libraries.
"""

import os
import numpy as np
import json
import pyvisgraph as vg
from shapely.geometry import LineString, Point
from shapely.ops import split, snap
from shapely import Polygon, to_geojson

import digger.constants as dconst


def point_to_shapely(point):
    """Converts (x, y) point to shapely point"""
    return Point(point[0], point[1])


def shapely_to_point(shapely_point):
    """Converts shapely point to (x, y) point"""
    return shapely_point.x, shapely_point.y


def move_by_angle_and_distance(point, angle, distance):
    """Returns new point after moving by angle and distance from given point
    """
    x, y = point
    x_new = x + distance * np.cos(np.deg2rad(angle))
    y_new = y + distance * np.sin(np.deg2rad(angle))
    return x_new, y_new


def get_angle_and_distance(initial_pos, target_pos):
    """Get angle and distance between two points.
    """
    dif = (target_pos[0] - initial_pos[0], target_pos[1] - initial_pos[1])
    angle = np.rad2deg(np.arctan2(dif[1], dif[0]))
    distance = np.sqrt(dif[0]**2 + dif[1]**2)
    return angle, distance


def convert_polygon_shapely_to_vg(polygon):
    """Converts shapely polygon to pyvisgraph polygon
    """
    points = json.loads(to_geojson(polygon))['coordinates'][0]
    polygon_vg = []
    for point in points:
        polygon_vg.append(vg.Point(point[0], point[1]))
    return polygon_vg


def get_shortest_path(start_point, end_point, obstacles):
    """Returns shortest path between start_point and end_point avoiding obstacles
    """
    # iterate overall obstacles and covert them to pyvisgraph polygons
    polygons = []
    for obstacle in obstacles:
        polygons.append(convert_polygon_shapely_to_vg(obstacle))

    # build the visibility graph
    workers = dconst.VISGRAPH_WORKERS
    graph = vg.VisGraph()
    # print("Starting building visibility graph")
    graph.build(polygons, workers=workers, status=False)
    # print("Finished building visibility graph")

    try:
        shortest_path = graph.shortest_path(
            vg.Point(start_point[0], start_point[1]),
            vg.Point(end_point[0], end_point[1]))
    except KeyError:
        # https://github.com/TaipanRex/pyvisgraph/issues/39
        # find a point not within the obstacle in proximity of the problematic point
        start_point_2 = find_point_outside_obstacles(obstacles, start_point)
        end_point_2 = find_point_outside_obstacles(obstacles, end_point)

        # case end point within the obstacle
        if end_point_2 is None:
            return LineString([Point(start_point), Point(start_point)])

        shortest_path = graph.shortest_path(
            vg.Point(start_point_2[0], start_point_2[1]),
            vg.Point(end_point_2[0], end_point_2[1]))

    # create a shapely line from the shortest path
    return LineString([(p.x, p.y) for p in shortest_path])


def find_point_outside_obstacles(obstacles, point, buffer_size=0.05):
    """Returns a point outside the obstacles in proximity of the given point
    Due to issue https://github.com/TaipanRex/pyvisgraph/issues/39
    """
    # create a buffer around the point
    buffer = Point(point).buffer(buffer_size, quad_segs=2)
    current_dist = 0
    res = None

    # iterate over the buffer points and return the first one outside the polygon
    for p in buffer.exterior.coords:
        tf_list = [obstacle.contains(Point(p)) for obstacle in obstacles]
        # print(tf_list)

        # return the most distant point
        if sum(tf_list) == 0:
            min_dist = min([obstacle.distance(Point(p)) for obstacle in obstacles])
            if min_dist > current_dist:
                current_dist = min_dist
                res = p
    return res


def limit_line(line, max_length, snap_tolerance: float = 1.0e-12):
    """Limit line length to max_length if it is longer
    """
    if line.length <= max_length:
        return line
    else:
        limit_point = line.interpolate(max_length)
        split_line = split(snap(line, limit_point, snap_tolerance), limit_point)
        return split_line.geoms[0]


def trim_line_to_obstacle(line, obstacles, buffer_size=1e-3):
    """Trims the last segment of line if within an obstacle
    """
    for obstacle in obstacles:
        # get difference line - exterior
        obstacle_ext = LineString(list(obstacle.exterior.coords))
        diff = line.difference(obstacle_ext)

        # case difference is empty - lies in the obstacle boundary
        if diff.is_empty:
            return line

        # make the split by polygon and get last segment
        split_segments = split(diff, obstacle)
        last_segment = split_segments.geoms[-1]

        # if last segment both end are within the
        first, last = last_segment.boundary.geoms

        # # error catching
        # try:
        #     first, last = last_segment.boundary.geoms
        # except ValueError:
        #     print(line)
        #     return line


        if obstacle.buffer(buffer_size).contains(first) & obstacle.buffer(buffer_size).contains(last):
            return line.difference(last_segment)
        else:
            pass
    return line


def get_unrestricted_std_moves(standard_moves_def, start_point, max_move):
    """Generate standard moves without avoiding obstacles"""
    return [move_by_angle_and_distance(start_point, angle,
                                       max_move * ratio) for angle, ratio in standard_moves_def]


def get_restricted_std_moves(standard_moves_def, max_move, start_point, obstacles):
    """Perform get_restricted_move for each end point in standard_moves_def
    """
    # generate all endpoints from definitions
    end_points = get_unrestricted_std_moves(standard_moves_def, start_point, max_move)

    # iterate over each end point
    shortest_paths = []
    end_points_final = []
    for end_point in end_points:

        # get restricted move
        restricted_end_point, move = get_restricted_move(start_point, end_point, obstacles, max_move)
        end_points_final.append(restricted_end_point)
        shortest_paths.append(move)

    return end_points_final, shortest_paths


def get_restricted_move(start_point, end_point, obstacles, max_move, get_path=False):
    """Generate move avoiding obstacles and limiting to max_move"""
    # get direct path
    direct_path = LineString([point_to_shapely(start_point), point_to_shapely(end_point)])

    # iterate over each obstacle
    hits_obstacle = False
    for obstacle in obstacles:

        # see if obstacle is hit
        obstacle_ext = LineString(list(obstacle.exterior.coords))
        intersections = obstacle_ext.intersection(direct_path)
        if not intersections.is_empty:
            hits_obstacle = True
            break

    # get the shortest path if obstacle is hit
    if hits_obstacle:
        shortest_path = get_shortest_path(start_point, end_point, obstacles)
        shortest_path = limit_line(shortest_path, max_move)
        result_path = trim_line_to_obstacle(shortest_path, obstacles)
    else:
        result_path = direct_path

    # in case the result is single point, convert it to line
    if len(result_path.boundary.geoms) > 0:
        res_end_point = shapely_to_point(result_path.boundary.geoms[1])
    else:
        res_end_point = start_point
        result_path = LineString([point_to_shapely(start_point), point_to_shapely(start_point)])

    if get_path:
        return res_end_point, result_path
    else:
        return res_end_point


def mm_to_inches(mm):
    return mm / 25.4


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_target_random_position(margin=3, y_range_share=0.3):
    """Get a random target position within the field with some margin"""
    # generator object
    rng = np.random.default_rng()

    # get ranges of x and y
    min_x = margin
    max_x = dconst.FIELD_X_RANGE - margin
    min_y = (1 - y_range_share) * dconst.FIELD_Y_RANGE
    max_y = dconst.FIELD_Y_RANGE - margin

    return rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)


def get_moving_target_start_end(margin=3, y_range_share=0.3):
    """Get a random start and end position for moving target objective"""
    # create shapely line from list of points
    min_x = margin
    max_x = dconst.FIELD_X_RANGE - margin
    min_y = (1 - y_range_share) * dconst.FIELD_Y_RANGE
    max_y = dconst.FIELD_Y_RANGE - margin
    points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
    line = LineString([point_to_shapely(point) for point in points])

    # get starting point
    rng = np.random.default_rng()
    start_ratio = rng.random()

    # get end from opposite side of rectangle
    if start_ratio > 0.5:
        end_ratio = rng.uniform(0, 0.5)
    else:
        end_ratio = rng.uniform(0.5, 1)

    # get start and end points
    start = line.interpolate(start_ratio, normalized=True)
    end = line.interpolate(end_ratio, normalized=True)

    return shapely_to_point(start), shapely_to_point(end)


