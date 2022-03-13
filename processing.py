import math

import numpy as np

import viz


def is_collinear(p1, p2, p3):
    return (p2[0] - p3[0]) * (p1[1] - p2[1]) == (p2[1] - p3[1]) * (p1[0] - p2[0])


def on_segment(p1, p2, p3):
    return (p1[0] < p3[0] < p2[0]) or (p1[0] > p3[0] > p2[0])


def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def norm_comp(x):
    return 1 if x > 0 else 0 if x == 0 else -1


def intersects(p1, p2, q1, q2):
    p1, q1, p2, q2 = p1, p2, q1, q2
    o1 = norm_comp(orientation(p1, q1, p2))
    o2 = norm_comp(orientation(p1, q1, q2))
    o3 = norm_comp(orientation(p2, q2, p1))
    o4 = norm_comp(orientation(p2, q2, q1))
    return (o1 != o2 and o3 != o4) or\
           (o1 == 0 and on_segment(p1, q1, p2)) or\
           (o2 == 0 and on_segment(p1, q1, q2)) or\
           (o3 == 0 and on_segment(p2, q2, p1)) or\
           (o4 == 0 and on_segment(p2, q2, q1))


def distance_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def gen_new_point(p1, p2, p3):
    # CO-LINEAR W/ P1 AND P2, PERPENDICULAR LINE AT NEW P1 INTERSECTS P3
    # THIS IS MAGIC DO NOT TOUCH!!!
    # https://www.wolframalpha.com/input/?i2d=true&i=f-k%3Dm%5C%2840%29j-l%5C%2841%29+and+f-n%3D-Divide%5B1%2Cm%5D%5C%2840%29j-o%5C%2841%29+for+f+and+j
    # Return value: j, f

    if p1[0] == p2[0]:  # Edge case: no division by 0
        return p1[0], p3[1]

    k = p1[1]
    l = p1[0]
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    n = p3[1]
    o = p3[0]

    f = (k + m * (-1 * l + m * n + o)) / (m * m + 1)
    j = (-1 * k * m + l * m * m + m * n + o) / (m * m + 1)

    return j, f


def gen_cutoff(c1, c2, i1, i2, i3):
    p1 = c1[i1]
    p2 = c1[i2]
    p3 = c2[i3]
    p = gen_new_point(p1, p2, p3)

    viz.show_contours_and_objects({
        "pts": c1,
        "fmt": "g-"
    }, {
        "pts": c2,
        "fmt": "g-"
    }, {
        "pts": [p1, p2],
        "fmt": "r-"
    }, {
        "pts": [p, p3],
        "fmt": "yo-"
    })

    if p1[0] < p[0] < p2[0] or p1[0] > p[0] > p2[0]:
        c1[i1] = p
    elif p1[0] < p2[0] < p[0] or p1[0] > p2[0] > p[0]:
        c1.pop(i1)
        return True
    return False


def get_distribution(c1, c2, pix_wid, ignore_more_than=None):
    # Copy c1 and c2 to not modify them
    c1 = c1[:]
    c2 = c2[:]
    og_c1 = c1[:]
    og_c2 = c2[:]
    # Make c1 and c2 parallel
    viz.show_contours_and_objects({
        "pts": c1,
        "fmt": "g-"
    }, {
        "pts": c2,
        "fmt": "g-"
    }, {
        "pts": [c1[0], c2[0]],
        "fmt": "bo-"
    }, {
        "pts": [c1[0], c2[-1]],
        "fmt": "bo-"
    })
    distance_first = distance_points(c1[0], c2[1])
    distance_last = distance_points(c1[0], c2[-1])
    if distance_last < distance_first:
        c2 = c2[::-1]

    distribution = {
        "total": 0,
        "ranges": []
    }
    while len(c1) > 2 and len(c2) > 2:
        if gen_cutoff(c1, c2, 0, 1, 0):
            continue
        if gen_cutoff(c1, c2, -1, -2, -1):
            continue
        if gen_cutoff(c2, c1, 0, 1, 0):
            continue
        if gen_cutoff(c2, c1, -1, -2, -1):
            continue
        break
    while len(c1) > 2 and len(c2) > 2:
        first_c1 = c1[0]
        first_c2 = c2[0]
        second_c1 = c1[1]
        second_c2 = c2[1]

        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": c1,
            "fmt": "g-"
        }, {
            "pts": c2,
            "fmt": "g-"
        }, {
            "pts": [first_c1, second_c1, second_c2, first_c2, first_c1],
            "fmt": "bo-"
        })

        potential_second_c1 = gen_new_point(first_c1, second_c1, second_c2)
        potential_second_c2 = gen_new_point(first_c2, second_c2, second_c1)

        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": c1,
            "fmt": "g-"
        }, {
            "pts": c2,
            "fmt": "g-"
        }, {
            "pts": [first_c1, second_c1, second_c2, first_c2, first_c1],
            "fmt": "bo-"
        }, {
            "pts": [second_c2, potential_second_c1],
            "fmt": "yo-"
        }, {
            "pts": [second_c1, potential_second_c2],
            "fmt": "yo-"
        })

        if on_segment(first_c1, second_c1, potential_second_c1):
            second_c1 = potential_second_c1
            c1[0] = second_c1
        else:
            c1.pop(0)
        if on_segment(first_c2, second_c2, potential_second_c2):
            second_c2 = potential_second_c2
            c2[0] = second_c2
        else:
            c2.pop(0)

        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": c1,
            "fmt": "g-"
        }, {
            "pts": c2,
            "fmt": "g-"
        }, {
            "pts": [first_c1, second_c1, second_c2, first_c2, first_c1],
            "fmt": "yo-"
        })
        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": og_c1,
            "fmt": "r-"
        }, {
            "pts": c2,
            "fmt": "g-"
        }, {
            "pts": [first_c1, first_c2],
            "fmt": "yo-"
        }, {
            "pts": [second_c1, second_c2],
            "fmt": "yo-"
        })

        skip_c1 = False
        for i in range(len(og_c1) - 1):
            if is_collinear(og_c1[i], og_c1[i + 1], first_c1) or is_collinear(og_c1[i], og_c1[i + 1], second_c1):
                continue
            if intersects(first_c1, first_c2, og_c1[i], og_c1[i + 1]) or \
                    intersects(second_c1, second_c2, og_c1[i], og_c1[i + 1]):
                skip_c1 = True
                break
        if skip_c1:
            c1.pop(0)
            continue
        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": c1,
            "fmt": "g-"
        }, {
            "pts": og_c2,
            "fmt": "r-"
        }, {
            "pts": [first_c1, first_c2],
            "fmt": "yo-"
        }, {
            "pts": [second_c1, second_c2],
            "fmt": "yo-"
        })
        skip_c2 = False
        for i in range(len(og_c2) - 1):
            if is_collinear(og_c2[i], og_c2[i + 1], first_c2) or is_collinear(og_c2[i], og_c2[i + 1], second_c2):
                continue
            if intersects(first_c1, first_c2, og_c2[i], og_c2[i + 1]) or \
                    intersects(second_c2, second_c2, og_c2[i], og_c2[i + 1]):
                skip_c2 = True
                break
        if skip_c2:
            c2.pop(0)
            continue

        weight = (distance_points(first_c1, second_c1) + distance_points(first_c2, second_c2)) / 2
        width_min = distance_points(first_c1, first_c2)
        width_max = distance_points(second_c1, second_c2)

        viz.show_contours_and_objects({
            "pts": og_c1,
            "fmt": "-"
        }, {
            "pts": og_c2,
            "fmt": "-"
        }, {
            "pts": c1,
            "fmt": "g-"
        }, {
            "pts": c2,
            "fmt": "g-"
        }, {
            "pts": [first_c1, second_c1],
            "fmt": "b-"
        }, {
            "pts": [first_c2, second_c2],
            "fmt": "b-"
        }, {
            "pts": [first_c1, first_c2],
            "fmt": "r-"
        }, {
            "pts": [second_c1, second_c2],
            "fmt": "r-"
        })
        if width_max < width_min:
            t = width_min
            width_min = width_max
            width_max = t
        weight *= pix_wid
        width_min *= pix_wid
        width_max *= pix_wid
        if ignore_more_than is not None and width_min > ignore_more_than:
            continue
        if ignore_more_than is not None and width_max > ignore_more_than:
            weight *= 1 - (width_max - ignore_more_than) / (width_max - width_min)
            width_max = ignore_more_than
        distribution["ranges"].append({
            "weight": weight,
            "min": width_min,
            "max": width_max
        })
        distribution["total"] += weight * pix_wid
    # Step 6: Return
    return distribution


def make_binned_xy(distribution, bin_width):
    all_mins = np.array([i["min"] for i in distribution["ranges"]])
    all_maxes = np.array([i["max"] for i in distribution["ranges"]])

    the_min = min(all_mins)
    the_max = max(all_maxes)

    the_min_floored = math.floor(the_min / bin_width) * bin_width
    the_max_ceiled = math.ceil(the_max / bin_width) * bin_width

    num_bins_necessary = int(round((the_max_ceiled - the_min_floored) / bin_width))

    all_x = []
    all_y = []

    for i in range(num_bins_necessary):
        x = the_min_floored + i * bin_width
        y = sum([
            j["weight"] / (j["max"] - j["min"])
            for j in distribution["ranges"]
            if j["min"] <= x <= j["max"]
        ])/distribution["total"]

        all_x.append(x)
        all_y.append(y)
    return all_x, all_y


def merge_dists(d1, d2):
    new_ranges = d1["ranges"][:]
    new_ranges.extend(d2["ranges"])
    return {
        "total": d1["total"] + d2["total"],
        "ranges": new_ranges
    }


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)
