from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyBigWig

SEQ_LEN = 196608 // 2  # = 98304


def get_bw_signal(bw_file, chrom, start, end):
    center = (start + end) // 2
    start = center - (SEQ_LEN // 2)
    end = center + (SEQ_LEN // 2)
    try:
        values = bw_file.values(chrom, start, end)
        values = np.nan_to_num(values).tolist()
    except:
        values = [np.nan] * SEQ_LEN
    return values


def avg_bin(array, n_bins):
    splitted = np.array_split(array, n_bins)
    return [np.mean(a) for a in splitted]


def get_chrom_sizes(base_path) -> dict:
    chrom_sizes = {}
    with open(base_path.joinpath("data/hg38.chrom.sizes"), "r") as f:
        for line in f:
            fields = line.split()
            chrom, size = fields[0], int(fields[1])
            if "_" not in chrom and 'EBV' not in chrom and 'chrM' not in chrom:
                chrom_sizes[chrom] = size
    return chrom_sizes


def random_region(chrom_sizes, bw_file):
    chrom = np.random.choice(list(chrom_sizes.keys()))
    start = np.random.randint(0, chrom_sizes[chrom] - SEQ_LEN)
    end = start + SEQ_LEN
    values = get_bw_signal(bw_file, chrom, start, end)
    return chrom, start, end, values


def main() -> None:
    base_path = Path(__file__).parents[1] # enformer-finetune/
    outdir = base_path.joinpath("data/sequences")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_csv(
        base_path.joinpath("data/master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt"),
        sep="\t",
    )

    bw_file = pyBigWig.open(str(base_path.joinpath("data/ENCFF972GVB.bw")))

    df = all_data[all_data['TAG'] == "PROMOTERS"]
    df['chrom'] = df['chrom'].astype(str)
    df[['start', 'end']] = df[['start', 'end']].astype(int)

    total = len(df)

    promoters = []
    for row in tqdm(df.itertuples(), total=total):

        values = get_bw_signal(bw_file, row.chrom, row.start, row.end)
        if np.any(np.isnan(values)):
            continue

        binned_values = avg_bin(values, n_bins=896)
        row_data = {
            'chrom': row.chrom,
            'start': row.start,
            'end': row.end,
            'values': binned_values
        }
        promoters.append(row_data)

    frac_null = 0.1
    len_null_values = int(len(promoters) * frac_null / (1 - frac_null))

    chrom_sizes = get_chrom_sizes(base_path)

    controls = []
    for i in tqdm(range(len_null_values)):
        find_control = True
        while find_control:
            chrom, start, end, values = random_region(chrom_sizes, bw_file)

            if np.any(np.isnan(values)):
                continue
            binned_values = avg_bin(values, n_bins=896)

            if np.all(np.array(binned_values) == 0):
                find_control = False
                row_data = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'values': binned_values
                }
                controls.append(row_data)

    promoters = pd.DataFrame(promoters)
    controls = pd.DataFrame(controls)

    promoters['seq_type'] = 'promoter'
    controls['seq_type'] = 'random'
    sequences = pd.concat((promoters, controls)).sample(frac=1)

    sequences[['chrom', 'start', 'end', 'seq_type', 'values']].to_csv(outdir.joinpath("promoter_dnase.bed"), sep="\t",
                                                                      header=False, index=False)


if __name__ == "__main__":
    main()
