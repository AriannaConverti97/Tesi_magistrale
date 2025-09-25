import pandas as pd
import numpy as np
import itertools

from tqdm import tqdm
from copy import deepcopy

from config import tv_bins, peep_bins, fio2_bins

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans

APACHE_RANGES = {
    "tempc": [(41, 4), (39, 3), (38.5, 1), (36, 0), (34, 1), (32, 2), (30, 3), (0, 4)],
    "meanbp": [(160, 4), (130, 3), (110, 2), (70, 0), (50, 2), (0, 4)],
    "heartrate": [(180, 4), (140, 3), (110, 2), (70, 0), (55, 2), (40, 3), (0, 4)],
    "ph": [(7.7, 4), (7.6, 3), (7.5, 1), (7.33, 0), (7.25, 2), (7.15, 3), (0, 4)],
    "sodium": [(180, 4), (160, 3), (155, 2), (150, 1), (130, 0), (120, 2), (111, 3), (0, 4)],
    "potassium": [(7, 4), (6, 3), (5.5, 1), (3.5, 0), (3, 1), (2.5, 2), (0, 4)],
    "creatinine": [(305, 4), (170, 3), (130, 2), (53, 0), (0, 2)],
    "wbc": [(40, 4), (20, 2), (15, 1), (3, 0), (1, 2), (0, 4)],
}

MAX_APACHE_SCORE = sum([l[-1][1] for l in APACHE_RANGES.values()]) + 12  # +12 is for max GCS score
INTERMEDIATE_REWARD_SCALING = 1
default_name=""

def to_discrete_action(actions):
    """
    Converts continuous ventilation actions (adjust_tidal_volume, fio2, peep)
    into discrete action indices based on predefined bins.

    Parameters:
        actions (array-like): A list or array of actions with shape (n, 3),
                              where each row contains values for
                              adjust_tidal_volume, fio2, and peep.

    Returns:
        np.ndarray: Array of discrete action indices corresponding to the input actions.

    Raises:
        ValueError: If any action is outside the defined bin ranges.
    """
    print("[INFO] Mapping all possible bin combinations to discrete action indices.")
    all_bin_indices = list(itertools.product(range(len(tv_bins)), range(len(fio2_bins)), range(len(peep_bins))))
    bin_map = {comb: idx for idx, comb in enumerate(all_bin_indices)}

    def find_bin(value, bins):
        for i, (low, high) in enumerate(bins):
            if low <= value < high:
                return i
        return None

    def discretize_row(row):
        tv_idx = find_bin(row['adjust_tidal_volume'], tv_bins)
        peep_idx= find_bin(row['peep'], peep_bins)
        fio2_idx= find_bin(row['fio2'], fio2_bins)

        if None in (tv_idx, peep_idx, fio2_idx):
            raise ValueError(f"Action (tv: {row['adjust_tidal_volume']}, fio2:{row['fio2']}, peep:{row['peep']}) not in action space")
        return bin_map[(tv_idx, fio2_idx, peep_idx)]


    action_df= pd.DataFrame(actions, columns=['adjust_tidal_volume', 'fio2', 'peep'])

    tqdm.pandas(desc="[INFO] Discretizing actions...")
    return np.array( action_df.progress_apply(discretize_row, axis=1) )

from config import tv_bins, fio2_bins, peep_bins,action_bin_definition
def to_discrete_action_bins(action_id, action_bin_definition=action_bin_definition):
    """
    Returns, for a given integer action_id, the corresponding bin indices for
    tv, fio2 and peep.
    """
    tv_range, fio2_range, peep_range = action_bin_definition[action_id]
    tv_bin = tv_bins.index(tv_range)
    fio2_bin = fio2_bins.index(fio2_range)
    peep_bin = peep_bins.index(peep_range)
    return tv_bin, fio2_bin, peep_bin


def to_action_ranges(action_id):
    """
    Returns, for a given action_id, the corresponding (min, max) tuples for tv,
    fio2 and peep settings.
    """
    tv_bin, fio2_bin, peep_bin = to_discrete_action_bins(action_id)
    return tv_bins[tv_bin], fio2_bins[fio2_bin], peep_bins[peep_bin]

def compute_apache2(row):
    score = 0

    for measurement, range_list in APACHE_RANGES.items():
        patient_value = row[measurement]
        for pair in range_list:
            if patient_value >= pair[0]:  # If this is the current range for this value
                score += pair[1]
                break

    score += (15 - row["gcs"])  # Different calculation of score for GCS

    return score

def build_trajectories(df, state_space, action_space, discretize_states = False, discretize_actions = True, n_cluster_states=660, intermidiate_reward=False, save=False, name=None):
    '''
    This assumes that the last timewise entry for a patient is the culmination of
     their episode or 72 hours from first row is culmination and that there exists a reward column
     which is always the reward of that episode

    *************IMPORTANT**********
    I assume that the dataframe has stay_id, hadm_id, subject_id, and charttime (start_time) as columns
    and it can be sorted by charttime (start_time) and end up in chronological order
    I also assume that there is a reward column in the dataset
    ********************************

    param df: the dataframe you want to build trajectories from
    param state_space.py: the columns of the daataframe you want to include in your state space
    '''
    binary_features = df.columns[df.apply(lambda x: set(x.unique()).issubset({0, 1}))]
    categorical_features = df.select_dtypes(include=['object']).columns
    numerical_features = list(set(df.columns) - set(binary_features) - set(categorical_features) - set(['subject_id', 'icu_day_start', 'peep','adjust_tidal_volume','fio2']))
    
    if intermidiate_reward:
        numerical_features += ["apache2", "apache2_shifted_up"]

    print("[INFO] Scaling numerical features.")
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print("[INFO] Scaling categorical features.")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[categorical_features] = encoder.fit_transform(df[categorical_features])

    
    # list of episodes that we'll poulate in the loop below
    episode_states = []

    # states prime, encoded list of states, sp[i]=the encoded rerpesentation of the ith state
    actions = []
    rewards = []
    states = []
    done_flags = []
    subject_ids=[]
    # loop through every unique value of the info tuple we had

    unique_infos = df['subject_id'].unique().tolist()
    
    for k in tqdm(range(len(unique_infos)), desc=f"[INFO] Building {name if name is not None else default_name } Trajectories for {len(unique_infos)} unique patients."): #n_cols=100
        i = unique_infos[k]
        
        # extract rows of the patient whos info we are iterating on
        episode_rows = df[df['subject_id'] == i].sort_values('icu_day_start')

        # instantiate the list of states, actions,... where the ith value in the list is the value at the ith timestep

        # iterate through rows which are sorted by charttime at the creation of episode_rows
        tdiff = episode_rows.iloc[0]['icu_day_start']
        for row in range(min(18, len(episode_rows))):
            end_index = min(18, len(episode_rows)) - 1

            # get the action, state, reward, next state, and whether or not the sequence is done in the current timestep
            state = episode_rows[state_space].iloc[row].values.tolist()

            action = episode_rows[action_space].iloc[row].values.tolist()

            if row == end_index:
                reward = 100 if episode_rows['hospmort90day'].iloc[row] == 0 else -100
            else:
                if not intermidiate_reward:
                    reward=0
                else:
                    reward = (-(
                                episode_rows['apache2'].iloc[row] - episode_rows['apache2_shifted_up'].iloc[
                            row])) / MAX_APACHE_SCORE
            dflag = 1 if row == end_index else 0

            # add the current time step info to the lists for this episode
            states.append(deepcopy(state))
            actions.append(deepcopy(action))
            rewards.append(deepcopy(reward))
            done_flags.append(deepcopy(dflag))
            subject_ids.append(i)

            # add to episodes a dictionairy with the mdp info for this episode
    print("[INFO] Trajectories built successfully.")
    actions = np.array(actions)
    rewards = np.array(rewards)
    done_flags = np.array(done_flags)
    states = np.array(states)
    state_space = np.array(state_space)
    subject_ids = np.array(subject_ids)
   
    info_dict = {
        "subject_id": subject_ids,
        "state_space": state_space,
        "actions": actions,
        "rewards": rewards,
        "done_flags": done_flags,
        "states": states
    }

    if discretize_states:
        print(f"[INFO] Discretizing states with KMeans [K = {n_cluster_states}]...")
        clusterer = KMeans(n_clusters=n_cluster_states)
        state_clusters = clusterer.fit_predict(info_dict['states'])
        info_dict['states_discretize']= np.array(state_clusters)
    
    if discretize_actions:
        print("[INFO] Discretize actions with predefine bins...")
        info_dict['actions_discretize']= to_discrete_action(info_dict['actions'])

    if save:
        print("[INFO] Save data files to 'Data/' folder...")
        np.save("data/states/raw_states.npy", info_dict['states'])
        np.save("data/states/state_space.npy", info_dict['state_space'])
        np.save("data/states/discrete_states.npy", info_dict['states_discretize'])
        np.save("data/rewards/rewards_with_intermidiate_fixed.npy", info_dict['rewards'])
        np.save("data/actions/3dactions_not_binned.npy",info_dict['actions'])
        np.save("data/actions/1dactions.npy", info_dict['actions_discretize'])
        np.save("data/done_flags/done_flags.npy", info_dict['done_flags'])
        np.save("data/indices/indices.npy", info_dict['subject_id'])
    # Return list of np arrays of states (one for each episode), because have to pass them through the LSTM autoencoder
    return info_dict