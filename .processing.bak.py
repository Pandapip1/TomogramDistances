import math

import cupy as cp


def distance_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def gen_new_point(p1, p2, p3):
    # CO-LINEAR W/ P1 AND P2, PERPENDICULAR LINE AT NEW P1 INTERSECTS P3
    # THIS IS MAGIC DO NOT TOUCH!!!
    # https://www.wolframalpha.com/input/?i2d=true&i=f-k%3Dm%5C%2840%29j-l%5C%2841%29+and+f-n%3D-Divide%5B1%2Cm%5D%5C%2840%29j-o%5C%2841%29+for+f+and+j
    if p1[0] == p2[0]:  # No division by 0
        return p1[0], p3[1]
    m1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
    new_p1 = [
        (-1 * p1[1] * m1 + p1[0] * m1 * m1 + m1 * p3[1] + p3[0]) / (m1 * m1 + 1),
        (p1[1] + m1 * (-1 * p1[0] + m1 * p3[1] + p3[0])) / (m1 * m1 + 1)
    ]
    return new_p1


def gen_cutoff(c1, c2, i1, i2, i3):
    p1 = c1[i1]
    p2 = c1[i2]
    p3 = c2[i3]
    p = gen_new_point(p1, p2, p3)
    if p1[0] < p[0] < p2[0] or p1[0] > p[0] > p2[0]:
        c1[i1] = p
    elif p1[0] < p2[0] < p[0] or p1[0] > p2[0] > p[0]:
        c1.pop(i1)
        return True
    return False


def shoelace(x_y):
    # https://stackoverflow.com/a/58515054/11628256
    x_y = cp.array(x_y)
    x_y = x_y.reshape(-1, 2)

    x = x_y[:, 0]
    y = x_y[:, 1]

    s1 = cp.sum(x * cp.roll(y, -1))
    s2 = cp.sum(y * cp.roll(x, -1))

    area = .5 * cp.absolute(s1 - s2)
    return area


def weighted_avg_and_variance(values, weights):
    # https://stackoverflow.com/a/2415343/11628256
    xp = cp.get_array_module(values)
    average = xp.average(values, weights=weights)
    variance = xp.average((values - average) ** 2, weights=weights)
    return average, variance


def get_variances(c1, c2, distances, weights):
    for i1 in range(len(c2) - 1):
        center_point = ((c2[i1][0] + c2[i1 + 1][0]) / 2, (c2[i1][1] + c2[i1 + 1][1]) / 2)
        weight = distance_points(c2[i1], c2[i1 + 1])
        smallest_distance = math.inf
        for i2 in range(len(c1) - 1):
            closest_point = gen_new_point(c1[i2], c1[i2 + 1], center_point)
            if (
                    c1[i2][0] < closest_point[0] < c1[i2 + 1][0] or
                    c1[i2][0] > closest_point[0] > c1[i2 + 1][0]
            ) and (
                    c1[i2][1] < closest_point[1] < c1[i2 + 1][1] or
                    c1[i2][1] > closest_point[1] > c1[i2 + 1][1]
            ):
                distance = distance_points(closest_point, center_point)
                if distance < smallest_distance:
                    smallest_distance = distance
        if smallest_distance != math.inf:
            distances.append(smallest_distance)
            weights.append(weight)


def get_min_max_distance_helper(c1, c2, true_distance):
    distances = []
    for i1 in range(len(c2) - 1):
        center_point = c2[i1]
        smallest_distance = math.inf
        for i2 in range(len(c1) - 1):
            closest_point = gen_new_point(c1[i2], c1[i2 + 1], center_point)
            reflection = gen_new_point(c2[i2], c1[i2 + 1], center_point)
            distance = distance_points(closest_point, center_point)
            if distance < smallest_distance:
                smallest_distance = distance
        if smallest_distance != math.inf:
            distances.append(smallest_distance)
    distances = sorted(distances)
    distances = [
        distances[i] for i in range(len(distances)-1)
        if all([distances[j+1]/distances[j] < 2 for j in range(i+1)])
    ]  # Cut off mistakes
    return distances[0], distances[1]


def get_min_max_distance(c1, c2):
    min_distance0, max_distance0 = get_min_max_distance_helper(c1, c2)
    min_distance1, max_distance1 = get_min_max_distance_helper(c2, c1)
    return min(min_distance0, min_distance1), max(max_distance0, max_distance1)


def get_distance_and_variance_between_contours(c1, c2):
    # Step 1: Order it correctly
    distance_first = distance_points(c1[0], c2[1])
    distance_last = distance_points(c1[0], c2[-1])
    if distance_last > distance_first:
        c2 = c2[::-1]  # Reverse it

    # Step 2: Cut off extras
    while True:
        if gen_cutoff(c1, c2, 0, 1, 0):
            continue
        if gen_cutoff(c1, c2, -1, -2, -1):
            continue
        if gen_cutoff(c2, c1, 0, 1, 0):
            continue
        if gen_cutoff(c2, c1, -1, -2, -1):
            continue
        break

    # Step 3: Calculate area and distance
    x_y = []
    x_y.extend(c1)
    x_y.extend(c2)
    area = shoelace(x_y)
    length = (sum([distance_points(c1[i + 1], c1[i]) for i in range(len(c1) - 1)]) +
              sum([distance_points(c2[i + 1], c2[i]) for i in range(len(c2) - 1)])) * 0.5
    distance = area / length

    # Step 4: Calculate min and max distance
    min_distance, max_distance = get_min_max_distance(c1, c2)

    # Step 5: Calculate variance
    distances = []
    weights = []
    get_variances(c1, c2, distances, weights)
    get_variances(c2, c1, distances, weights)
    # For some reason sometimes the distances go sky high. Skip those!
    weights = [weights[i] for i in range(len(weights)) if distances[i] < distance * 2]
    distances = [distances[i] for i in range(len(distances)) if distances[i] < distance * 2]
    _, variance = weighted_avg_and_variance(cp.array(distances), cp.array(weights))

    # Step 6: Return
    return distance, variance, min_distance, max_distance
