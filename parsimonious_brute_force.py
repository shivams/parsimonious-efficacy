#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from z3 import Bool, And, Implies, Solver, sat, is_true
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import random
import sys
import signal
import itertools
from scipy.spatial import ConvexHull

def plot_points(p, q, r, t):
    points = np.array([p, q, r, t])
    labels = ['p', 'q', 'r', 't']
    
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], color='red')
    plt.plot([p[0], q[0]], [p[1], q[1]], 'bo-', label='p-q')
    plt.plot([q[0], r[0]], [q[1], r[1]], 'go-', label='q-r')
    plt.plot([r[0], p[0]], [r[1], p[1]], 'mo-', label='r-p')
    plt.scatter(t[0], t[1], color='blue', marker='x', s=100, label='t')
    
    # Add labels to the points
    for i, label in enumerate(labels):
        plt.text(points[i, 0], points[i, 1], f'{label}', fontsize=12, ha='right')
    
    plt.legend()
    plt.title('Points p, q, r, and t')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    #  plt.show()

    # save the 2D plot as a PNG image
    plt.savefig('2D_plot.png')


def save_plot(points, deduction_count, deduced_relations, split=False):
    def save_single_plot(points, deduction_count, timestamp):
        labels = [f'{i}' for i in range(len(points))]
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], color='red')
        for i, label in enumerate(labels):
            plt.text(points[i, 0], points[i, 1], label, fontsize=12, ha='right')
        
        # Plot deduced triangles with different colors
        for relation in deduced_relations:
            if relation.deduced:
                p, q, r = relation.points
                p, q, r = np.array(p), np.array(q), np.array(r)
                color = np.random.rand(3,)  # Random color
                plt.plot([p[0], q[0]], [p[1], q[1]], linestyle='--', color=color)
                plt.plot([q[0], r[0]], [q[1], r[1]], linestyle='--', color=color)
                plt.plot([r[0], p[0]], [r[1], p[1]], linestyle='--', color=color)
        
        plt.title(f'Configuration with {deduction_count} Deductions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        
        # Ensure the "figures" directory exists
        if not os.path.exists("figures"):
            os.makedirs("figures")
        
        point_count = len(points)
        filename = os.path.join("figures", f"{point_count}_{deduction_count}_{timestamp}.png")
        
        # Check if file exists and append a random number if it does
        if os.path.exists(filename):
            random_number = random.randint(1000, 9999)
            filename = os.path.join("figures", f"{point_count}_{deduction_count}_{timestamp}_{random_number}.png")
        
        plt.savefig(filename)
        plt.close()

    def save_split_plots(points, deduction_count, timestamp):
        labels = [f'{i}' for i in range(len(points))]
        for index, relation in enumerate(deduced_relations):
            if relation.deduced:
                plt.figure()
                plt.scatter(points[:, 0], points[:, 1], color='red')
                for i, label in enumerate(labels):
                    plt.text(points[i, 0], points[i, 1], label, fontsize=12, ha='right')
                
                # Plot a single deduced triangle with a specific color
                p, q, r = relation.points
                p, q, r = np.array(p), np.array(q), np.array(r)
                color = np.random.rand(3,)  # Random color
                plt.plot([p[0], q[0]], [p[1], q[1]], linestyle='--', color=color)
                plt.plot([q[0], r[0]], [q[1], r[1]], linestyle='--', color=color)
                plt.plot([r[0], p[0]], [r[1], p[1]], linestyle='--', color=color)
                
                plt.title(f'Configuration with {deduction_count} Deductions - Triangle {index + 1}')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.grid(True)
                
                # Ensure the "figures" directory exists
                if not os.path.exists("figures"):
                    os.makedirs("figures")
                
                point_count = len(points)
                filename = os.path.join("figures", f"{point_count}_{deduction_count}_{timestamp}_triangle_{index + 1}.png")
                
                # Check if file exists and append a random number if it does
                if os.path.exists(filename):
                    random_number = random.randint(1000, 9999)
                    filename = os.path.join("figures", f"{point_count}_{deduction_count}_{timestamp}_triangle_{index + 1}_{random_number}.png")
                
                plt.savefig(filename)
                plt.close()
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if split:
        save_split_plots(points, deduction_count, timestamp)
    else:
        save_single_plot(points, deduction_count, timestamp)

class CCRelation:
    def __init__(self, p, q, r, is_ccw, deduced=False):
        # Convert numpy arrays to tuples to ensure they are hashable
        self.points = tuple(sorted([tuple(p), tuple(q), tuple(r)]))
        self.is_ccw = is_ccw
        self.deduced = deduced
        self.relations = {}
        self.add_relation(p, q, r, is_ccw)

    def __hash__(self):
        return hash(self.points)

    def __eq__(self, other):
        return self.points == other.points

    def _cyclic_permutations(self, p, q, r):
        p, q, r = tuple(p), tuple(q), tuple(r)
        return [(p, q, r), (q, r, p), (r, p, q)]

    def _anti_permutations(self, p, q, r):
        p, q, r = tuple(p), tuple(q), tuple(r)
        return [(p, r, q), (r, q, p), (q, p, r)]

    def add_relation(self, p, q, r, is_ccw):
        p, q, r = tuple(p), tuple(q), tuple(r)
        # Add the relation with cyclic permutations
        for triplet in self._cyclic_permutations(p, q, r):
            self.relations[triplet] = is_ccw

        # Add the anti-symmetry relation with anti-cyclic permutations
        for triplet in self._anti_permutations(p, q, r):
            self.relations[triplet] = not is_ccw

    def get_relation(self, p, q, r):
        return self.relations.get((tuple(p), tuple(q), tuple(r)), None)

def generate_random_points(num_points, lower_bound=0, upper_bound=100):
    return np.random.uniform(lower_bound, upper_bound, (num_points, 2))

def generate_random_points_circle(n):
    '''generates random points all on a circle
    our total deductions should be zero for such case since all points lie on convex hull
    '''
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += np.random.uniform(0, 2 * np.pi / n, n)
    points = np.c_[np.cos(angles), np.sin(angles)]
    return points


def generate_random_points_concentric(num_points, center=(0, 0), radius_step=1):
    points = []
    for i in range(num_points):
        radius = (i + 1) * radius_step
        angle = np.random.uniform(0, 2 * np.pi)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    return np.array(points)


def generate_random_points_triplets_concentric(random_seed, num_points, center=(0, 0), radius_step=1):

    np.random.seed(random_seed)
    RANDOM_SEED = random_seed

    def generate_three_points_on_circle(center, radius):
        #  print(f"Our random seed is f{RANDOM_SEED}")
        np.random.seed(RANDOM_SEED)
        #  angles = np.random.uniform(0, 2 * np.pi, 3)
        angles = []
        min_angle_separation = np.pi/3
        while len(angles) < 3:
            angle = np.random.uniform(0, 2 * np.pi)
            if all(min(abs(angle - a), 2 * np.pi - abs(angle - a)) >= min_angle_separation for a in angles):
                angles.append(angle)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append((x, y))
        return np.array(points)

    def find_incircle(points):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        A, B, C = hull_points
        a = np.linalg.norm(B - C)
        b = np.linalg.norm(A - C)
        c = np.linalg.norm(A - B)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        radius = area / s
        center = (a * A + b * B + c * C) / (a + b + c)
        return center, radius

    points = []
    current_center = np.array(center)
    current_radius = radius_step

    num_triplets = num_points // 3

    for _ in range(num_triplets):
        triplet_points = generate_three_points_on_circle(current_center, current_radius)
        points.extend(triplet_points)

        # Calculate the incircle of the current convex hull
        incircle_center, incircle_radius = find_incircle(np.array(triplet_points))

        # Update the center and radius for the next set of points
        current_center = incircle_center
        current_radius = incircle_radius * 0.9  # Slightly less than the incircle's radius

    # If there are leftover points to be generated, place them on the last incircle
    if num_points % 3 != 0:
        remaining_points = num_points % 3
        leftover_points = generate_three_points_on_circle(current_center, current_radius)[:remaining_points]
        points.extend(leftover_points)

    points = np.array(points)

    #  # Plot the points
    #  plt.figure()
    #  plt.scatter(points[:, 0], points[:, 1], color='red')
    #  plt.title(f'Generated {num_points} Points in Triplets Concentric')
    #  plt.xlabel('X-axis')
    #  plt.ylabel('Y-axis')
    #  plt.grid(True)
    #  plt.show()

    return points

def ccw(p, q, r):
    # Returns true if the points p, q, r are in counter-clockwise order
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) > 0

def deduce_cc_relation(p, q, r, relations):
    solver = Solver()
    for t_relation in relations:
        t = t_relation.points[0]
        if t == tuple(p) or t == tuple(q) or t == tuple(r):
            continue
        tqr_key = CCRelation(t, q, r, None)
        ptr_key = CCRelation(p, t, r, None)
        pqt_key = CCRelation(p, q, t, None)

        if tqr_key in relations and ptr_key in relations and pqt_key in relations:
            #  import ipdb; ipdb.set_trace()
            tqr = Bool(f"{tqr_key.points}")
            ptr = Bool(f"{ptr_key.points}")
            pqt = Bool(f"{pqt_key.points}")
            pqr = Bool(f"{CCRelation(p, q, r, None).points}")

            solver.push()
            solver.add(tqr == bool(next(x for x in relations if x == tqr_key).get_relation(t, q, r)),
                       ptr == bool(next(x for x in relations if x == ptr_key).get_relation(p, t, r)),
                       pqt == bool(next(x for x in relations if x == pqt_key).get_relation(p, q, t)))
            solver.add(Implies(And(tqr, ptr, pqt), pqr))

            # plot points p, q, r, t
            #  plot_points(p, q, r, t)
            #  import ipdb; ipdb.set_trace()

            if solver.check() == sat:
                model = solver.model()
                # Convert the Z3 boolean expression to a Python boolean
                #  import ipdb; ipdb.set_trace()
                if is_true(model[pqr]):
                    return True
                else:
                    return None
            solver.pop()
    return None

def hybrid_cc_relations(points):
    num_points = len(points)
    relations = set()

    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                p, q, r = points[i], points[j], points[k]
                deduced_relation = deduce_cc_relation(p, q, r, relations)

                if deduced_relation is None:
                    is_ccw = ccw(p, q, r)
                    new_relation = CCRelation(p, q, r, is_ccw, deduced=False)
                else:
                    new_relation = CCRelation(p, q, r, deduced_relation, deduced=True)
                relations.add(new_relation)

    return relations


def count_deductions(relations):
    return sum(1 for relation in relations if relation.deduced)

# Main function to find maximum deductions
def find_max_deductions():
    max_deductions = 0
    auto_increment = 1

    while True:
        points = generate_random_points(9)
        relations = hybrid_cc_relations(points)
        deduction_count = count_deductions(relations)
        print(f"Deductions for this set of points: {deduction_count}")

        if deduction_count >= max_deductions:
            max_deductions = deduction_count
            save_plot(points, deduction_count, auto_increment, split=True)
            auto_increment += 1
            print(f"New max deductions: {deduction_count}")



def run_hybrid_approach(auto_increment, num_points, pattern_type="triplets"):

    start_time = time.time()

    # Set a unique random seed for each process
    np.random.seed(auto_increment*42 + int(datetime.now().timestamp()))
    seed = auto_increment*42000 + int(datetime.now().timestamp())
    #  print(f"The seed that we set is {seed}")
    #  print(f"Our auto_increment is {auto_increment}")

    if pattern_type == "concentric":
        points = generate_random_points_concentric(num_points, center=(0, 0), radius_step=1)
    elif pattern_type == "triplets":
        points = generate_random_points_triplets_concentric(seed, num_points, center=(0, 0), radius_step=1)
    else:
        points = generate_random_points(num_points)  # Default random point generation

    max_deductions = 0
    best_relations = None
    best_deduced_relations = None

    #  # Iterate over each possible sequence of points
    #  for permuted_points in itertools.permutations(points):
    #      relations = hybrid_cc_relations(np.array(permuted_points))
    #      deduction_count = count_deductions(relations)
    #
    #      if deduction_count > max_deductions:
    #          max_deductions = deduction_count
    #          best_relations = relations
    #          best_deduced_relations = [relation for relation in relations if relation.deduced]

    # Generate all combinations of triplets
    triplets = list(itertools.combinations(points, 3))

    # Iterate over each possible sequence of triplets
    for permuted_triplets in itertools.permutations(triplets):
        # Flatten the list of triplets to a list of points
        permuted_points = [point for triplet in permuted_triplets for point in triplet]
        permuted_points_array = np.array(permuted_points)
        relations = hybrid_cc_relations(permuted_points_array)
        deduction_count = count_deductions(relations)

        if deduction_count > max_deductions:
            max_deductions = deduction_count
            best_deduced_relations = [relation for relation in relations if relation.deduced]

    hybrid_time = time.time() - start_time
    print(f"Time taken for checking all sequences for given point set: {hybrid_time:.6f} seconds")

    return max_deductions, points, best_deduced_relations, auto_increment


def find_max_deductions_parallel(num_points, pattern_type="random"):
    max_deductions = 0
    auto_increment = 1
    num_cores = os.cpu_count() - 4  # Use fewer cores than the maximum
    split = True  # Change to False if you want a single plot file
    MIN_DEDUCTIONS_DESIRED = 0

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit initial jobs with unique auto_increment values
        futures = {executor.submit(run_hybrid_approach, auto_increment + i, num_points, pattern_type): auto_increment + i for i in range(num_cores)}

        def shutdown_handler(signum, frame):
            print("Shutting down gracefully...")
            executor.shutdown(wait=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)

        while True:
            print(f"Current max deductions: {max_deductions}")
            try:
                for future in as_completed(futures):
                    deduction_count, points, deduced_relations, inc = future.result()

                    if deduction_count > max_deductions:
                        max_deductions = deduction_count
                        print(f"New max deductions: {deduction_count}")
                        if deduction_count >= MIN_DEDUCTIONS_DESIRED:
                            save_plot(points, deduction_count, deduced_relations, split=split)

                    # Increment the auto_increment for the new job
                    auto_increment = max(futures.values()) + 1
                    futures[executor.submit(run_hybrid_approach, auto_increment, num_points, pattern_type)] = auto_increment
            except KeyboardInterrupt:
                shutdown_handler(None, None)


def main():

    while True:
        points = generate_random_points_circle(6)
        #  points = generate_random_points(10)
        relations = hybrid_cc_relations(points)

        # Checking some relations
        deduced, numerical = 0, 0
        for relation in relations:
            source = "deduced" if relation.deduced else "numerically calculated"
            #  print(f"{relation.points}: {relation.is_ccw}, {source}")
            if relation.deduced:
                deduced += 1
            else:
                numerical += 1

        print(f"Deduced: {deduced}, Numerical: {numerical}")




if __name__ == '__main__':

    #  find_max_deductions()
    find_max_deductions_parallel(num_points=6, pattern_type="triplets")

    #  main()



