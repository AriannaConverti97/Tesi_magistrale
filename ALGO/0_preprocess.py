import numpy as np
import pandas as pd
import argparse
import os 

from tqdm import tqdm
from utils.preprocessing_function import impute_variables, take_first_event_and_weaning
from config import col_to_remove, limit, ffill_windows_clinical, demographics

import warnings
warnings.filterwarnings('ignore')


def remove_outliers(df, range=limit):
    print("[INFO] Remove the outliers.")
    for var in range:
        lower, upper = range[var]
        df.loc[(df[var]< lower) | (df[var]> upper), var]=np.nan
    return df

def sample_and_hold(df, ffill_windows_clinical=ffill_windows_clinical):
    print("[INFO] Apply Sample and Hold (SAH) with the clinical limit.")
    for var in ffill_windows_clinical.keys():
        df[var]= df.groupby('subject_id')[var].ffill(limit=ffill_windows_clinical[var])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("This script reads a CSV file containing patient data, processes it by computing IBW,"
                                                    "removing outliers, imputing variables, and extracting the first ventilation episodes."
                                                    "Optionally, the final processed dataset can be saved to a CSV file."))
    parser.add_argument('--name', dest='file_name', type=str, default="Data/dati.csv",
                        help='The path in which read files (default is Data/dati.csv)')
    parser.add_argument('--out', dest='output_dir', type=str, default="Data/",
                        help='Directory in which to output (default is Data/)')
    parser.add_argument('--save', dest='save', type=bool, default=False,
                        help="If passed, don't save the final dataset.")

    args = parser.parse_args()
    assert args.file_name is not None, "Specify the file name using the command --name='filename.csv'"
    if args.output_dir is None:
        output_dir = "Data/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    print("[INFO] Reading csv file and constructing the DataFrame.")
    df = pd.read_csv(args.file_name, low_memory=False)
    assert df is not None, "The CSV reading failed."
    len_df = df.shape[0]
    
    df['ibw']= np.where(
        df['gender']=='M',
        50 + 0.91 * (df['height'] - 152.4),
        45.5 + 0.91 * (df['height'] - 152.4)
    )

    df.drop(col_to_remove, axis=1, inplace=True)

    df = remove_outliers(df)
    assert df.shape[0] == len_df, "Remove outliers changes dimension of dataFrame"
    df= sample_and_hold(df)
    assert df.shape[0] == len_df, "Sample and Hold changes dimension of dataFrame"

    df= impute_variables(df, demographics)
    assert df.shape[0] == len_df, "Impute variables changes dimension of dataFrame"

    tqdm.pandas(desc="[INFO] Extracting the first ventilation episodes for each patient", ncols=100)
    df= df.groupby('subject_id').progress_apply(take_first_event_and_weaning).reset_index(drop=True)
    
    assert df.shape[0] <= 18* len(df.groupby("subject_id").first().reset_index()), "take_first_event changes the dimension of DataFrame"
    print(f"[INFO] Change dimension of the dataset from {len_df} to {df.shape[0]}")
    assert (df.groupby('subject_id').size()<19).all(), "Each subject must have fewer than 19 rows"
    assert (df.groupby('subject_id')['weaning_success'].nunique()==1).all(), "'weaning_success' must be consistent (unique) per subject"
    assert all(df.notna().all()), "DataFrame contains missing (NaN) values."
    
    if args.save:
        print(f"[INFO] Save the dataset in 'Data/dati_imputati.csv'.")
        if "/" in output_dir:
            file_name= f"{output_dir}dati_imputati.csv"
        else:
            file_name= f"{output_dir}/dati_imputati.csv"
        df.to_csv(file_name, index=False)

    print("[INFO] Preprocessing completed successfully.")
