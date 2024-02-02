import functools
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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


def plot_metric(
    baselines,
    l2os,
    metric,
    show_max_iters=None,
    log_metric=False,
    with_err_bars=False,
    conv_window=None,
    save_fig_to_path=None,
):
    ### plot comparison
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ### baseline optimizers
    for baseline_name, baseline_dict in baselines.items():
        opter_metric = np.array(baseline_dict["log"][metric])
        plot_config = baseline_dict["plot_config"] if "plot_config" in baseline_dict else None
        
        if show_max_iters is None:
            x = range(opter_metric.shape[1])
            y = np.mean(opter_metric, axis=0)
        else:
            x = range(opter_metric[..., :show_max_iters].shape[1])
            y = np.mean(opter_metric[..., :show_max_iters], axis=0)
        if conv_window and conv_window > 1:
            y_removed_start = y[:conv_window - 1]
            y = np.convolve(y, np.ones(conv_window), "valid") / conv_window
            y = np.concatenate([y_removed_start, y])
        
        if plot_config:
            sns.lineplot(
                x=x,
                y=y,
                label=baseline_name,
                ax=ax,
                **plot_config["main"],
            )
        else:
            sns.lineplot(
                x=x,
                y=y,
                label=baseline_name,
                ax=ax,
            )

        if with_err_bars:
            if show_max_iters is None:
                y_std = np.std(opter_metric, axis=0)
            else:
                y_std = np.std(opter_metric[..., :show_max_iters], axis=0)
            if conv_window and conv_window > 1:
                y_removed_start = y_std[:conv_window - 1]
                y_std = np.convolve(y_std, np.ones(conv_window), "valid") / conv_window
                y_std = np.concatenate([y_removed_start, y_std])
            if plot_config and plot_config["err_bars"]:
                sns.lineplot(
                    x=x,
                    y=y_std,
                    ax=ax,
                    **plot_config["err_bars"],
                )
            else:
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.2,
                )
            

    ### L2O optimizers
    for l2o_name, l2o_dict in l2os.items():
        l2o_metric = np.array(l2o_dict["log"][metric])
        plot_config = l2o_dict["plot_config"] if "plot_config" in l2o_dict else None

        if show_max_iters is None:
            x = range(l2o_metric.shape[1])
            y = np.mean(l2o_metric, axis=0)
        else:
            x = range(l2o_metric[:,:show_max_iters].shape[1])
            y = np.mean(l2o_metric[:,:show_max_iters], axis=0)
        if conv_window and conv_window > 1:
            y_removed_start = y[:conv_window - 1]
            y = np.convolve(y, np.ones(conv_window), "valid") / conv_window
            y = np.concatenate([y_removed_start, y])
        if plot_config:
            sns.lineplot(
                x=x,
                y=y,
                label=fr"{l2o_name}",
                ax=ax,
                **plot_config["main"],
            )
        else:
            sns.lineplot(
                x=x,
                y=y,
                label=fr"{l2o_name}",
                ax=ax,
            )

        if with_err_bars:
            if show_max_iters is None:
                y_std = np.std(l2o_metric, axis=0)
            else:
                y_std = np.std(l2o_metric[:,:show_max_iters], axis=0)
            if conv_window and conv_window > 1:
                y_removed_start = y_std[:conv_window - 1]
                y_std = np.convolve(y_std, np.ones(conv_window), "valid") / conv_window
                y_std = np.concatenate([y_removed_start, y_std])
            if plot_config and plot_config["err_bars"]:
                sns.lineplot(
                    x=x,
                    y=y_std,
                    ax=ax,
                    **plot_config["err_bars"],
                )
            else:
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.2,
                )

    ### plot settings
    # y-label
    ax.set_xlabel("Iteration")
    if metric == "loss":
        metric_as_label = "Loss"
    else:
        metric_as_label = metric
    if log_metric:
        metric_as_label = f"{metric_as_label} (log scale)"
    ax.set_ylabel(metric_as_label)

    # scale of y-axis
    if log_metric:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, None)

    # legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    legend = ax.get_legend()
    for legend_handle in legend.legendHandles:
        legend_handle.set_linewidth(3.0)

    # x-ticks
    x_ticks = ax.get_xticks()
    x_ticks = np.linspace(0, len(x), 3)
    ax.set_xticks(x_ticks)

    ### y-ticks
    if log_metric:
        y_ticks = ax.get_yticks()
        # y_ticks = np.linspace(y_ticks[1], y_ticks[-2], 2)
        y_ticks = [y_ticks[idx] for idx in range(1, len(y_ticks) - 1, len(y_ticks) // 3)]
        ax.set_yticks(y_ticks)

        # ax.set_ylim(0.185, 2.8)
        # ax.set_yticks([0.3, 1, 2])
        # ax.set_yticklabels([0.3, 1, 2])

        # ax.set_yticks([0.2, 1, 2])
        # ax.set_yticklabels([0.2, 1, 2])
    else:
        y_max = ax.get_ylim()[1]
        y_max = np.ceil(y_max / 0.5) * 0.5 # round
        y_ticks = np.linspace(0, y_max, 3)
        ax.set_yticks(y_ticks)

    # axins.set_xlim(0, 80)
    # axins.set_ylim(0.35, 2.5)
    # # axins.set_yscale("log")
    # axins.set_xticks([0, 40, 80])
    # # axins.set_yticks([1.])
    # axins.set_yticklabels([1.0], fontsize=7)
    # axins.set_xticklabels([0, 40, 80], fontsize=7.5, position=(0., .08))
    # ax.indicate_inset_zoom(axins)
        
    ### remove top and right axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ### remove legend's frame
    legend = ax.get_legend()
    legend.set_frame_on(False)

    # Add tick lines for both major and minor ticks
    ax.tick_params(which="major", length=3, bottom=True, left=True)
    ax.tick_params(which="minor", length=3, bottom=False, left=False)

    plt.show()

    ### save the figure
    if save_fig_to_path is not None:
        fig.savefig(save_fig_to_path, bbox_inches="tight")