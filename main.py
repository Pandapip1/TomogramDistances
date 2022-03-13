import math
import os
import shutil
import subprocess
import warnings

import numpy as np
from matplotlib import pyplot as plt

import processing

data_directory = "./data"
temp_directory = "./temp"


def run():
    xs = {}
    ys = {}
    for files in os.listdir(temp_directory):
        path = os.path.join(temp_directory, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    for filename in os.listdir(data_directory):
        spec_name = '.'.join(filename.split('.')[:-1])
        if not any([filename.endswith(f".{x}") for x in ["imod"]]):
            continue
        in_file = f"{data_directory}/{filename}"
        out_file = f"{temp_directory}/{spec_name}.txt"
        subprocess.check_output(f"model2point -c -ze -int {in_file} {out_file}", shell=True)
    cell_dimensions = {}
    model_files = ['.'.join(filename0.split('.')[:-1]) for filename0 in os.listdir(temp_directory)]
    for filename in os.listdir(data_directory):
        spec_name = '.'.join(filename.split('.')[:-1])
        if not any([filename.endswith(f".{x}") for x in ["mrc", "st"]]):
            continue

        def is_float(f):
            try:
                float(f)
                return True
            except ValueError:
                return False
        cella_str = subprocess.check_output(f"header -p {data_directory}/{filename}", shell=True).decode("utf-8").split(" ")
        cella = [float(x) for x in cella_str if is_float(x)]
        if cella[0] != cella[1]:
            warnings.warn(f"{spec_name} has non-constant dimensions")
            continue
        cell_dimensions[spec_name] = cella[0] / 10  # angstroms to nm
        if spec_name not in model_files:
            warnings.warn(f"{spec_name} has no model file")
    del model_files
    fig = plt.figure()
    plots_dim = int(math.ceil(math.sqrt(len(os.listdir(temp_directory)))))
    plots_i = 1
    for filename in os.listdir(temp_directory):
        spec_name = '.'.join(filename.split('.')[:-1])
        if spec_name not in cell_dimensions.keys():
            warnings.warn(f"{spec_name} has no tilt series")
            continue
        with open(f"{temp_directory}/{filename}") as contour_file:
            raw_contours = [[int(y) for y in x.split(" ") if y != ""] for x in contour_file.readlines()]
            contour_pairs = {}
            for raw_contour in raw_contours:
                if raw_contour[3] not in contour_pairs.keys():
                    contour_pairs[raw_contour[3]] = {}
                if raw_contour[0] not in contour_pairs[raw_contour[3]].keys():
                    contour_pairs[raw_contour[3]][raw_contour[0]] = []
                contour_pairs[raw_contour[3]][raw_contour[0]].append([raw_contour[1], raw_contour[2]])
            final_contour_pairs = []
            for z in contour_pairs.keys():
                if len(contour_pairs[z].keys()) != 2:
                    warnings.warn(f"{spec_name} doesn't have 2 contours on z = {z}")
                    continue
                pair = []
                for c in contour_pairs[z].keys():
                    pair.append(contour_pairs[z][c])
                final_contour_pairs.append(pair)
            if len(final_contour_pairs) == 0:
                warnings.warn(f"{spec_name} has no pairs of contours.")
                continue
            # Generate distribution
            dist = {
                "total": 0,
                "ranges": []
            }
            for contour_pair in final_contour_pairs:
                dist = processing.merge_dists(dist, processing.get_distribution(
                    c1=contour_pair[0],
                    c2=contour_pair[1],
                    pix_wid=cell_dimensions[spec_name],
                    ignore_more_than=None
                ))
            # Generate (x, y) pairs
            step = 0.05
            x, y = processing.make_binned_xy(dist, step)
            # Generate cutoffs
            ignore_thresh = 0.0065
            left = 0
            right = -1
            while y[left] < ignore_thresh:
                left += 1
            while y[right] < ignore_thresh:
                right -= 1
            # Plot
            ax = fig.add_subplot(plots_dim, plots_dim, plots_i)
            ax.bar(x, y)
            ax.set_xlim(left=x[left], right=x[right])
            ax.set_title(f"{spec_name}")
            ax.set_xlabel("Distance (nm)")
            ax.set_ylabel("P")
            plots_i += 1
            xs[spec_name] = x
            ys[spec_name] = y
    fig.suptitle("Probability Distribution of Inter-Membrane Distances")
    fig.tight_layout(w_pad=0.75, h_pad=0.9)
    fig.savefig(f"output/Figure.png")
    plt.show()
    # Gen stats
    all_widths = {}
    all_stds = {}
    all_mins = {}
    all_maxes = {}
    for spec_name in xs.keys():
        x = np.array(xs[spec_name])
        y = np.array(ys[spec_name])
        ignore_thresh = 0.0065
        left = 0
        right = -1
        while y[left] < ignore_thresh:
            left += 1
        while y[right] < ignore_thresh:
            right -= 1
        all_mins[spec_name], all_maxes[spec_name] = x[left], x[right]
        all_widths[spec_name], all_stds[spec_name] = processing.weighted_avg_and_std(x[left:right], y[left:right])

    print(f"Name\t\tWidth\t\tStd Deviation\tMinimum Dist\tMaximum Dist")
    for spec_name in all_widths.keys():
        width = all_widths[spec_name]
        std = all_stds[spec_name]
        min_d = all_mins[spec_name]
        max_d = all_maxes[spec_name]
        if len(spec_name) < 8:
            spec_name += "\t"
        print(f"{spec_name[:15]}\t{width:.2f} nm\t{std:.2f} nm\t\t{min_d:.2f} nm\t{max_d:.2f} nm")


if __name__ == '__main__':
    run()
