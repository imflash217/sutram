import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline


def use_svg_display():
    """
    Use the SVG format to display plot in jupyter
    """
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the figure size for matplotlib
    """
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def has_one_axis(x):
    return (
        hasattr(x, "ndim")
        and x.ndim == 1
        or isinstance(x, list)
        and not hasattr(x[0], "__len__")
    )


def plot(
    xb,
    yb=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot data points"""
    ## step-1: check for the LEGEND
    if legend is None:
        legend = []

    ## step-2: set the FIGSIZE
    set_figsize(figsize)

    ## step-3: check axes
    axes = axes if axes else plt.gca()

    ## step-4: check "x"
    if has_one_axis(xb):
        xb = [xb]

    ## step-5: check "y"
    if yb is None:
        xb, yb = [[]] * len(xb), xb
    elif has_one_axis(yb):
        yb = [yb]

    ## step-6: align x & y
    if len(xb) != len(yb):
        xb = xb * len(yb)

    ## step-7:
    axes.cla()

    ## step-8
    for x, y, fmt in zip(xb, yb, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)

    ## step-9:
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
