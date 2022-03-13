from matplotlib import pyplot as plt

num = 0


def show_contours_and_objects(*args, title=""):
    global num
    return
    ax = plt.gca()
    for obj in args:
        pts = obj["pts"]
        fmt = obj["fmt"]
        ax.plot([i[0] for i in pts], [i[1] for i in pts], fmt)
    ax.set_title(title)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    plt.savefig(f"vizes/{num}.png")
    num += 1

    plt.clf()

