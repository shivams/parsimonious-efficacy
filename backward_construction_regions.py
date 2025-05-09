#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ((2024-06-10)) This script is used for calculating in how many triangles will a 
# point be located, as we add it to different regions within a space of n points all of which 
# are located in a convex hull.

import numpy as np
import scipy.spatial
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Point, Polygon as ShapelyPolygon
import time

def generate_random_points(n):
    start_time = time.time()
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += np.random.uniform(0, 2 * np.pi / n, n)
    points = np.c_[np.cos(angles), np.sin(angles)]
    end_time = time.time()
    #  print(f"Generating random points took {end_time - start_time} seconds")
    return points

def compute_convex_hull(points):
    start_time = time.time()
    hull = scipy.spatial.ConvexHull(points)
    end_time = time.time()
    #  print(f"Computing convex hull took {end_time - start_time} seconds")
    return hull

def generate_triangles(points, hull):
    triangles = []
    for i, j, k in itertools.combinations(hull.vertices, 3):
        triangle = ShapelyPolygon([points[i], points[j], points[k]])
        triangles.append(triangle)
    return triangles

def point_in_triangle(point, triangle):
    return triangle.contains(point)

def on_click(event, points, triangles):
    start_time = time.time()
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    clicked_point = Point(x, y)
    count = sum(1 for triangle in triangles if point_in_triangle(clicked_point, triangle))

    print(f'Point ({x:.2f}, {y:.2f}) is inside {count} triangles.')

    # Highlight the triangles that the point lies in
    ax.cla()  # Clear the axes
    plot_triangles(points, hull, triangles)
    for triangle in triangles:
        if point_in_triangle(clicked_point, triangle):
            coords = np.array(triangle.exterior.coords)
            polygon = Polygon(coords, closed=True, fill=True, edgecolor='red', alpha=0.3)
            ax.add_patch(polygon)

    # Plot the clicked point
    ax.plot(x, y, 'rx')

    # Display the count on the figure
    ax.text(0.05, 0.95, f'Count: {count}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    plt.draw()

    end_time = time.time()
    print(f"On click time: {end_time - start_time:.6f} seconds")

def plot_triangles(points, hull, triangles):
    ax.plot(points[:, 0], points[:, 1], 'o', label='Points')
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-')
    for triangle in triangles:
        coords = np.array(triangle.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], 'b--')

n = 6
print(f"Generating random points for n = {n}")
start_time = time.time()

points = generate_random_points(n)
hull = compute_convex_hull(points)
triangles = generate_triangles(points, hull)

fig, ax = plt.subplots()
plot_triangles(points, hull, triangles)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.6f} seconds")

cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, triangles))
plt.show()
