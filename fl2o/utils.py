import functools
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def plot_log(
    runs,
    only_metrics=None,
    log_metrics=None,
    conv_win=1,
    min_max_y_config=None,
    save_to=None,
):
    """
    Plots all metrics of all runs in log.
    """
    run_names = list(runs.keys())
    runs = {k: {
        "log": deepcopy(v["log"]),
        "plot_config": v["plot_config"] if "plot_config" in v else None,
    } for k, v in runs.items()}
    if only_metrics is not None:
        for run_name in run_names:
            for metric in list(runs[run_name]["log"].keys()):
                if metric not in only_metrics:
                    del runs[run_name]["log"][metric]
    n_axes = len(runs[run_names[0]]["log"])

    fig, ax = plt.subplots(
        int(np.ceil(n_axes/2)),
        2,
        figsize=(18, 4.5 + 5.5 * n_axes/2),
        facecolor="w",
    )
    ax = ax.flatten()
    if n_axes % 2 == 1:
        ax[-1].axis("off")

    for i, run_name in enumerate(run_names):
        curr_log = runs[run_name]["log"]
        metrics = list(curr_log.keys())
        
        for j, metric in enumerate(metrics):  ## curr_log[metric] of shape (n_tests, n_iters)
            ### prepare what to plot
            y = np.array(curr_log[metric])

            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            if conv_win and conv_win > 1:
                ### conv mean
                y_removed_start = y_mean[:conv_win - 1]
                y_mean = np.convolve(y_mean, np.ones(conv_win)/conv_win, mode="valid")
                y_mean = np.concatenate([y_removed_start, y_mean])

                ### conv std
                y_std_removed_start = y_std[:conv_win - 1]
                y_std = np.convolve(y_std, np.ones(conv_win)/conv_win, mode="valid")
                y_std = np.concatenate([y_std_removed_start, y_std])

            ### plot mean
            if "plot_config" in runs[run_name] and runs[run_name]["plot_config"] is not None:
                ax[j].plot(y_mean, **runs[run_name]["plot_config"], label=run_name)
            else:
                ax[j].plot(y_mean, label=run_name)
            
            ### plot std
            ax[j].fill_between(
                np.arange(len(y_mean)),
                y_mean - y_std,
                y_mean + y_std,
                alpha=0.15,
                color=ax[j].get_lines()[-1].get_color(),
            )

            ### design
            ax[j].legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, fontsize=14, frameon=False)
            ax[j].set_xlabel("iteration", fontsize=14)
            ax[j].set_ylabel(metric, fontsize=14)
            ax[j].tick_params(axis="both", which="major", labelsize=12)
            ax[j].tick_params(axis="both", which="minor", labelsize=12)
            if log_metrics is not None and metric in log_metrics:
                ax[j].set_yscale("log")
            
            ### remove top and right spines
            ax[j].spines["top"].set_visible(False)
            ax[j].spines["right"].set_visible(False)

            ### set legend linewidth
            legend = ax[j].get_legend()
            for legend_handle in legend.legendHandles:
                legend_handle.set_linewidth(3.)
            
            ### set min max y
            if min_max_y_config is not None and metric in min_max_y_config:
                ax[j].set_ylim(min_max_y_config[metric])
    
    plt.tight_layout(h_pad=2.5)

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight", h_pad=2.5)
    
    plt.show()


def plotter(to_plot, run_log, meta_training_log, run_num, **kwargs):
    plt.plot(run_log[to_plot]),
    plt.title(f"{to_plot} (run {run_num})"),
    plt.show()
