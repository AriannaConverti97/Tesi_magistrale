import pandas as pd
import numpy as np
from miceforest import ImputationKernel
from tqdm import tqdm

def print_missing_pct(dataframe, variables):
    """
    Formatted print of the number of NaNs in `dataframe` for each variable in `variables`.
    """
    ljust = max(map(len, variables))
    for var in variables:
        if round(dataframe[var].isna().mean()*100,2)>0:
            print('{} {} % \t ({})'.format(var.ljust(ljust), round(dataframe[var].isna().mean()*100,2), dataframe[var].isna().sum() ))
        else: 
            print('{} {} %'.format(var.ljust(ljust), round(dataframe[var].isna().mean()*100,2)))


def compute_pao2_fio2(df):
    return round(df['pao2']/df['fio2'],2)

def compute_shock_index(df):
    return round(df['heartrate']/df['sysbp'],2)

def compute_meanbp(df):
    return round((df['sysbp']+2*df['diasbp'])/3)

def compute_adjust_tidal_volume(df):
    return round(df['tidal_volume']/df['ibw'],2)


def recompute_variables(df):
    print("\t[INFO] Computing PaO2/FiO2 ratio.")
    df['pao2fio2ratio']= compute_pao2_fio2(df)

    print("\t[INFO] Computing shock index.")
    df['shock_index'] = compute_shock_index(df)

    print("\t[INFO] Computing mean blood pressure (meanbp).")
    df['meanbp'] = compute_meanbp(df)

    print("\t[INFO] Computing adjusted tidal volume.")
    df['adjust_tidal_volume']= compute_adjust_tidal_volume(df)
    return df

def impute_categorial_variables(df):
    """
    Imputes and preprocesses categorical variables in the ICU dataset.

    This function performs the following steps:
    1. Converts continuous infusion rates of vasoactive drugs into binary indicators (0/1), 
       dropping the original columns.
    2. Fills missing values in 'admission_category' with the default category 
       'Monitoraggio Postoperatorio'.
    3. Handles missing values in categorical variables based on conditional flags:
       - If no infection, transfusion, or sedation is indicated (flag == 0), missing types 
         are filled with 'None'.
       - If sedation is indicated (is_sedated == 1), missing 'sedation_type' is filled with 
         'Propofol'.
    4. Fills missing values in 'vent_type' within each subject using forward/backward fill.
       If all values for a subject are missing, imputes based on the most frequent 
       'vent_type' associated with the subject's modal 'vent_mode'.
    5. Cleans 'vent_mode' for intubated patients:
       - Invalid combinations of 'Standby' and 'Ambient' are set to NaN.
       - Remaining missing values are filled using forward/backward fill within subject.

    Parameters:
        df (pd.DataFrame): The input ICU dataset with clinical and ventilation-related features.

    Returns:
        pd.DataFrame: The modified DataFrame with imputed categorical variables and cleaned structure.

    Raises:
        AssertionError: If any imputed columns still contain missing values after processing.
    """
    print("[INFO] Starting categorical variable imputation.")
    # Imputation VASOPRESSOR
    vaso = ['rate_dobutamine', 'rate_dopamine', 'rate_epinephrine', 'rate_norepinephrine', 'rate_vasopressin']
    for col in vaso:
        new_str = col.replace('rate', 'has')
        df[new_str] = (df[col]>0).astype(int)
        print(f"\t[INFO] Created binary column '{new_str}' from '{col}'.")
        df = df.drop(col, axis=1)
        assert df[new_str].isna().mean()==0, f"Column '{new_str}' contains missing values (NaNs)."
    
    # Imputation ADMISSION CATEGORY
    df['admission_category'].fillna("Monitoraggio Postoperatorio", inplace=True)
    print(f"\t[INFO] Filled missing values in 'admission_category' with 'Monitoraggio Postoperatorio'")
    assert df['admission_category'].isna().mean()==0, ""

    # Imputation INFECTION, SEDATION, TRANSFUSION
    categories = ['infection_at_icu_admission', 'transfusion_24h', 'is_sedated']
    types = ['site_of_infection', 'transfusion_type', 'sedation_type']
    for cat, type in zip(categories, types):
        if df.loc[df[cat]==0, type].isna().mean()!=0:
            print(f"\t[INFO] Filling NaNs in '{type}' where '{cat}' == 0 with 'None'")
            df.loc[df[cat]==0, type] = (df.loc[df[cat]==0, type].fillna('None'))
        
        if cat=='is_sedated':
            print(f"\t[INFO] Filling NaNs in '{type}' where '{cat}' == 1 with 'Propofol'")
            df.loc[df[cat]==1, type] = (df.loc[df[cat]==1, type].fillna('Propofol'))
        
        assert df.loc[df[cat]==0, type].isna().mean()==0, f"Missing values found in '{type}' when '{cat}' equals 0"
        assert df.loc[df[cat]==1, type].isna().mean()==0, f"Missing values found in '{type}' when '{cat}' equals 1"

    # Imputation VENT_TYPE
    df['vent_type']= df.groupby('subject_id')['vent_type'].transform(lambda x: x.ffill().bfill())
    print("\t[INFO] Applied forward/backward fill to 'vent_type' grouped by 'subject_id'")
    print(f"\t[INFO] Imputing missing 'vent_type' for each subject with value based on most frequent mapping")

    has_no_null = df.groupby('subject_id')['vent_type'].apply(lambda x: x.notna().any())
    ct = pd.crosstab(df['vent_mode'], df['vent_type'])
    for idx in has_no_null[~has_no_null].index.tolist():
        mask = df['subject_id']==idx
        val=df.loc[mask, 'vent_mode'].mode().iloc[0]
        val_to_impute= ct.loc[val].dropna().idxmax()
        df.loc[mask, 'vent_type']= df.loc[mask, 'vent_type'].fillna(val_to_impute)
    assert df['vent_type'].isna().mean()==0, "'vent_type' column contains missing values (NaNs)"

    # Imputation VENT_MODE
    mask_vent = (df['is_intubated']==1)
    mask_stb = (df['vent_mode']=='Standby')
    mask_amb = (df['vent_mode']=='Ambient')

    print("\t[INFO] Imputing 'vent_mode' for intubated patients and remove Standby and Ambient values.")
    df.loc[mask_vent & mask_stb & mask_amb, 'vent_mode']= np.nan
    df.loc[mask_vent, 'vent_mode'] = df.loc[mask_vent, 'vent_mode'].ffill().bfill()
    assert df.loc[mask_vent, 'vent_mode'].isna().mean()==0, "Missing values detected in 'vent_mode' where 'mask_vent' is True. Please handle NaNs before proceeding."

    print("[INFO] Categorical variable imputation completed.")
    return df

def impute_variables(df, demographics_vars, random_state=1997):
    """
    Performs a two-stage imputation on a clinical ICU dataset using MICE via `impyute`.

    The function follows these steps:
    
    1. Categorical Imputation:
        - Calls `impute_categorial_variables` to handle and fill missing values in key 
          categorical columns, including ventilation-related variables.
        - Converts all non-numeric columns to categorical dtype.

    2. Static Imputation (First stage):
        - Extracts static demographic variables (e.g., age, gender) per patient using the 
          first record per 'subject_id'.
        - Applies MICE (Multiple Imputation by Chained Equations) to impute missing static 
          values.
        - Merges the imputed static values back into the main DataFrame.

    3. Longitudinal Imputation (Second stage):
        - Removes columns not to be imputed ('icu_day_start', 'vent_mode').
        - Applies MICE to the full dataset for time-varying variables.

    4. Finalization:
        - Concatenates back the untouched time-related columns.
        - Recomputes derived or dependent variables via `recompute_variables`.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing clinical data with potential missing values.
        demographics_vars (List[str]): List of static variables to impute separately.
        random_state (int): Random seed for reproducibility during MICE.

    Returns:
        pd.DataFrame: The fully imputed DataFrame, ready for downstream analysis or modeling.

    Raises:
        AssertionError: If any column expected to be categorical fails conversion.
    """
    # Imputation CATEGORICAL VARIABLES
    df = impute_categorial_variables(df)
    
    print("[INFO] Converting non-numeric columns to categorical type.")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].astype("category")
            assert not pd.api.types.is_numeric_dtype(df[col]), f"The columns {col} could not be converted to categorical."
    
    # Imputation of static data for each subject_id
    print("[INFO] Starting imputation.")
    patients = df.groupby("subject_id")[demographics_vars].first().reset_index()

    kds_static= ImputationKernel(patients, random_state=random_state)

    for _ in tqdm(range(11), desc="\t[MICE] iterations (static)", ncols=100):
        kds_static.mice(1)
    imputed_patients = kds_static.complete_data()

    df = df.drop(columns=demographics_vars, errors = "ignore")
    df = df.merge(imputed_patients, on ="subject_id", how="left")

    # Subsequently, impute the missing variables
    kds_long= ImputationKernel(
        df.drop(columns=['icu_day_start', 'vent_mode'], axis=1),
        random_state=random_state
    )
    
    for _ in tqdm(range(11), desc="\t[MICE] iterations (longitudinal)", ncols=100):
        kds_long.mice(1)
    df_final = kds_long.complete_data()

    print("[INFO] Reattaching time-based columns ('icu_day_start', 'vent_mode')")
    df_final = pd.concat([df_final, df[['icu_day_start', 'vent_mode']]], axis=1)

    # Recompute some variables after imputation 
    print("[INFO] Recomputing derived variables.")
    df_final = recompute_variables(df_final)
    
    print("[INFO] Imputation pipeline finished successfully!")
    return df_final

def take_first_event_and_weaning(group, max_len=18):
    """
    Extracts the first ventilation episode for a patient.

    Parameters:
    - group: DataFrame containing all time records of a patient.
    - max_len: Maximum number of row to include in the episode. 
               = 18, we take the first 72 hours
    
    Returns:
    - DataFrame with rows of the first ventilation episode,
        plus a 'weaning_success' column.
        Returns empty DataFrame if no intubation is found.
    """
    intubated_indices = group[group['is_intubated']==1].index
    first_index = intubated_indices.min()
    consecutive_count=0
    last_index=None

    for idx in intubated_indices:
        if last_index is None or idx==last_index+1:
            consecutive_count +=1
        else:
            break
        last_index=idx
    
    window= group.loc[first_index:last_index].copy()

    end_window = min(group.index.max(), last_index + 18) # 18: If I don't receive a refund within 72 hours
    future_segment= group.loc[last_index+1: end_window]
    weaning_success = 0 if (future_segment['is_intubated']==1).any() else 1

    window = window.assign(weaning_success=weaning_success)
    return window.head(max_len)
