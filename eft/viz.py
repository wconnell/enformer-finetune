import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import kipoiseq
import numpy as np
import seaborn as sns


def fancy_plot_tracks(tracks, interval, height=2.0, stats=None):
    n_tracks = len(tracks)

    # Adjust height ratios for the plots and the title
    gridspec_settings = {'height_ratios': [0.1] + [1]*n_tracks} if stats else {'height_ratios': [1]*n_tracks}

    fig, axes = plt.subplots(n_tracks + (1 if stats else 0), 1, figsize=(20, height * n_tracks), gridspec_kw=gridspec_settings, sharex=True)

    # If rho is provided, set it as the title for the additional axis and hide the axis
    if stats:
        rho, pval = stats
        axes[0].set_title(f'PEARSONR: {rho:.4f}, PVALUE: {pval:.4e}', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        track_axes = axes[1:]
    else:
        track_axes = axes

    for ax, (title, y) in zip(track_axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    track_axes[-1].set_xlabel(str(interval))
    plt.tight_layout(h_pad=2.0)  # Adjust the horizontal padding between subplots
    return fig

def save_plot_to_pdf(pdf, target, pred, loc, stats):
    """Save the given tracks to the specified PDF."""
    target_interval = kipoiseq.Interval(*loc)
    tracks = {
        'TRUE': target,
        'PREDICTED': pred
    }
    fancy_plot_tracks(tracks, target_interval, stats=stats)
    pdf.savefig()  # save the current figure into a pdf page
    plt.close()