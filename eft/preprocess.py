import sys
import argparse
from pathlib import Path
import typing
import numpy as np
import pandas as pd
import pyBigWig

SEQ_LEN = 196608 // 2

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
    binned_array = [np.mean(a) for a in splitted]
    return binned_array

def main(args) -> None:
    all_data = pd.read_csv(args.outdir / "master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt", 
                           sep="\t", 
                           dtype={'chrom': str, 'start': int, 'end': int}
                           )

    bw_file = pyBigWig.open("data/ENCFF972GVB.bw")

    # write out segments
    for i in ['PROMOTERS', 'training', 'validation', 'test']:
        df = all_data[all_data['TAG']==i]

        seq_values = []
        for row in df.itertuples():

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
            seq_values.append(row_data)

    seq_values = pd.DataFrame(seq_values)

    seq_values.to_pickle(args.outdir / f"{i}.pkl")
    seq_values[['chrom', 'start', 'end', 'values']].to_csv(args.outdir / f"{i}.bed", sep="\t", header=False, index=False)



if __name__ == "__main__":
    """
    Parse Arguments
    
    """
    desc = "Description"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--out", type=Path, help="Directory to write data.")
    args = parser.parse_args()
    sys.exit(main(args))








