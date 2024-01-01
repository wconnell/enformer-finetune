import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from scipy.stats import pearsonr
import numpy as np
import kipoiseq
import seaborn as sns


def fancy_plot_tracks(tracks, interval, height=2.0, stats=None):
    n_tracks = len(tracks)

    gridspec_settings = {'height_ratios': [0.1] + [1] * n_tracks} if stats else {'height_ratios': [1] * n_tracks}

    fig, axes = plt.subplots(n_tracks + (1 if stats else 0), 1, figsize=(20, height * n_tracks),
                             gridspec_kw=gridspec_settings, sharex=True)

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


def process_data(seq_dl, target_dl, promoter_values_df, save_path):
    """
        TODO: (a part of) this function needs to be integrated into models.py on_epoch_end() function.
    """

    """Process the data and save the plots to PDFs based on the value of rho."""

    # Create new PDF files for saving plots
    with PdfPages(save_path.joinpath('rho_less_0.3.pdf')) as pdf_below, PdfPages(
            save_path.joinpath('rho_greater_0.3.pdf')) as pdf_above:

        # No gradient computation needed
        with torch.no_grad():
            for idx, (seq_batch, (target_batch,)) in enumerate(zip(seq_dl, target_dl)):

                # Predictions
                seq_batch = seq_batch.to(device)
                out = enformer(seq_batch).cpu().squeeze().numpy()
                target_batch = target_batch.squeeze().cpu().numpy()

                # Iterate through each index of the batch
                for i in range(len(target_batch)):
                    stats = pearsonr(target_batch[i], out[i])
                    rho, pval = stats

                    # Decide which PDF to save to based on rho
                    if rho < 0.3:
                        save_plot_to_pdf(pdf_below, target_batch[i], out[i],
                                         promoter_values_df.loc[idx, ['chrom', 'start', 'end']].tolist(), stats)
                    elif rho > 0.3:
                        save_plot_to_pdf(pdf_above, target_batch[i], out[i],
                                         promoter_values_df.loc[idx, ['chrom', 'start', 'end']].tolist(), stats)


# Example call to the main processing function
save_path = Path("drive/MyDrive/colab/enformer/2023-10-26_05-09-47/")
process_data(seq_dl, target_dl, promoter_values_df, save_path)
