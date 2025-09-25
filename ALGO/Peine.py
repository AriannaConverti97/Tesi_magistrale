
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from Models.MDPtoolbox import mdp_policy_iteration_with_Q

# from Models.common import compute_physician_policy
from sklearn.cluster import KMeans
import pandas as pd
from config import alive_state, died_state, n_states, n_actions, state_space, action_space, n_cluster_states
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.compute_trajectory_discretize import build_trajectories, compute_apache2, to_action_ranges
import warnings
warnings.filterwarnings('ignore')


# class Peine:
#     def __init__(self, n_cluster_states=660, n_actions=7**3, gamma=0.99, reward_val=100, 
#                  transition_threshold=5, soften_factor=0.01, random_state=None):
#         self.n_cluster_states = n_cluster_states
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.random_state = random_state
#         self.reward_val = reward_val
#         self.transition_threshold = transition_threshold
#         self.soften_factor = soften_factor
#         self.n_states = self.n_cluster_states + 2
#         self.absorbing_states = [self.n_cluster_states + 1, self.n_cluster_states] # absorbing state numbers
#         self.rewards = [self.reward_val, -self.reward_val]
        
#         self.clusterer = None
#         self.Q = None
#         self.physician_policy = None
#         self.transitionr = None
#         self.R = None
    
#     def train(self, train_df, val_df=None):        
#         physpol, transitionr, R = compute_physician_policy(
#             train_df,
#             self.n_states,
#             self.n_actions,
#             self.absorbing_states,
#             reward_val=self.reward_val,
#             transition_threshold=self.transition_threshold
#         )

#         print("Policy iteration")
#         self.Q = mdp_policy_iteration_with_Q(
#             np.swapaxes(transitionr, 0, 1),
#             R,
#             self.gamma,
#             np.ones(self.n_states)
#         )[-1]
#         self.R = R
#         self.physician_policy = physpol
#         self.transitionr = transitionr

def create_sansr(data):
    data['next_state']= data['states_discretize'].shift(-1).ffill().astype(int)
    data.loc[(data['done_flags']==1) & (data['rewards']==1), 'states_discretize']=alive_state
    data.loc[(data['done_flags']==1) & (data['rewards']==-1), 'states_discretize']=died_state


    data.loc[(data['done_flags']==1) & (data['rewards']==1), 'next_state']=alive_state
    data.loc[(data['done_flags']==1) & (data['rewards']==-1), 'next_state']=died_state

    val =0
    ep = []
    
    for _, valore in enumerate(data['done_flags']):
        ep.append(val)
        if valore ==1:
            val +=1
    
    data['ep']=ep
    return data

def create_success_faileure_episode_idx(data):
    ep_idx_succes= data[data['rewards']==1]['ep']
    ep_idx_faileure = data[data['rewards']==-1]['ep']
    return ep_idx_succes, ep_idx_faileure

def peine_mc_iterate(snsasr, Qn, gamma, n_epochs=1, learning_rate=0.1, unsafety_prob=0.0, safety_map=None):
    """
    Monte-carlo-based iteration of the training procedure according to tabular FQI & Peine's supplementary discussion.
    
    snsas: numpy ndarray with discretized state-nextstate-action tuples
    r: a function that returns the immediate reward for a state-action pair
    Qn: dictionary that maps iteration indices to Qn-estimates
    n: iteration number
    gamma: discount factor
    n_epochs: number of times to iterate over dataset
    learning rate: learning rate alpha
    """
    def epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map):
        for _, (s, ns, a, er) in enumerate(snsasr):
            #print(s,a, ns)
            if unsafety_prob == 1.0:
                # We do not care about the safety rules
                Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns),:]) - Qn[s,a])
            elif unsafety_prob == 0.0:
                if safety_map[a]:
                    Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns), safety_map]) - Qn[s,a])
                else:
                    # taken action not safe, disregard sample
                    pass
            else:
                raise ValueError("Only unsafety probs in {0.0, 1.0} supported for now")
        return Qn
   
    for _ in tqdm(range(n_epochs), "Training Peine "):
        #print(f"Epoch {n+1}")
        Qn = epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map)
        assert np.nanmax(Qn) < 100, "Scores > 100 should not occur, found: {}".format(np.nanargmax(Qn))
    return Qn
def peine_mc_iterate(snsasr, Qn, gamma, n_epochs=1, learning_rate=0.1, unsafety_prob=0.0, safety_map=None):
    """
    Monte-carlo-based iteration of the training procedure according to tabular FQI & Peine's supplementary discussion.
    
    snsas: numpy ndarray with discretized state-nextstate-action tuples
    r: a function that returns the immediate reward for a state-action pair
    Qn: dictionary that maps iteration indices to Qn-estimates
    n: iteration number
    gamma: discount factor
    n_epochs: number of times to iterate over dataset
    learning rate: learning rate alpha
    """
    def epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map):
        for _, (s, ns, a, er) in enumerate(snsasr):
            #print(s,a, ns)
            if unsafety_prob == 1.0:
                # We do not care about the safety rules
                Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns),:]) - Qn[s,a])
            elif unsafety_prob == 0.0:
                if safety_map[a]:
                    Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns), safety_map]) - Qn[s,a])
                else:
                    # taken action not safe, disregard sample
                    pass
            else:
                raise ValueError("Only unsafety probs in {0.0, 1.0} supported for now")
        return Qn
   
    for _ in tqdm(range(n_epochs), "Training Peine "):
        #print(f"Epoch {n+1}")
        Qn = epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map)
        assert np.nanmax(Qn) < 100, "Scores > 100 should not occur, found: {}".format(np.nanargmax(Qn))
    return Qn
import torch
from tqdm import tqdm

def peine_mc_iterate_torch(snsasr, Qn, gamma, n_epochs=1, learning_rate=0.1, unsafety_prob=0.0, safety_map=None, device="cuda"):
    """
    Monte-Carlo-based iteration according to tabular FQI & Peine.

    Args:
        snsasr: torch.Tensor of shape (n_samples, 4) -> [s, ns, a, reward]
        Qn: torch.Tensor of shape (n_states, n_actions) - initial Q-table
        gamma: discount factor
        n_epochs: number of iterations over dataset
        learning_rate: alpha
        unsafety_prob: 0.0 or 1.0
        safety_map: torch.BoolTensor of shape (n_actions,) or None
        device: "cpu" or "cuda"
    Returns:
        Updated Qn tensor
    """
    snsasr = torch.tensor(snsasr, dtype=torch.float32)
    snsasr = snsasr.to(device)
    Qn = torch.tensor(Qn, dtype=torch.float32).to(device)
    
    for epoch_idx in tqdm(range(n_epochs), desc="Training Peine"):
        for s, ns, a, r in snsasr:
            s, ns, a, r = int(s.item()), int(ns.item()), int(a.item()), r.item()
            
            if unsafety_prob == 1.0:
                Qn[s, a] = Qn[s, a] + learning_rate * (r + gamma * torch.max(Qn[ns, :]) - Qn[s, a])
            
            elif unsafety_prob == 0.0:
                if safety_map is not None and safety_map[a]:
                    safe_indices = torch.nonzero(safety_map).squeeze(-1)
                    Qn[s, a] = Qn[s, a] + learning_rate * (r + gamma * torch.max(Qn[ns, safe_indices]) - Qn[s, a])
                else:
                    # action not safe, skip update
                    pass
            else:
                raise ValueError("Only unsafety_prob in {0.0, 1.0} supported")

        # Optional sanity check
        assert torch.max(Qn) < 100, f"Scores > 100 should not occur, found max: {torch.max(Qn)}"

    return Qn
    
if __name__ == "__main__":
    import os
    discretize_actions = True
    discretize_states = True
    intermidiate_reward =True
    plot=False

    discretize_actions_name = "_ACT_DIS"
    discretize_states_name = "_STATE_DIS_"
    default_name= ""
    intermidiate_reward_name = "_INT_REW"

    name_file = f"Data/traj{discretize_actions_name if discretize_actions else default_name}{discretize_states_name if discretize_states else default_name}{n_cluster_states if discretize_actions else default_name}{intermidiate_reward_name if intermidiate_reward else default_name}.csv"

    if os.path.isfile(name_file):
        data = pd.read_csv(name_file)
    else:
        data = pd.read_csv("Data/dati_imputati.csv", keep_default_na=False, na_values=np.nan)
        assert data.notna().all().all(), "The DataFrame contains NaN values."

        data['apache2'] = data.apply(compute_apache2, axis=1)
        data['apache2_shifted_up'] = data['apache2'].shift(-1)

        MDP_dataset_dict = build_trajectories(data, state_space, action_space, discretize_states = discretize_states, discretize_actions = discretize_actions, n_cluster_states=n_cluster_states, save=False, intermidiate_reward=intermidiate_reward)
        data = pd.DataFrame({k: MDP_dataset_dict[k] for k in ['subject_id','states_discretize','actions_discretize', 'rewards','done_flags']})

        if data['rewards'].max()>1:
            data['rewards']= data['rewards']/100

        data['next_state']= data.groupby('subject_id')['states_discretize'].shift(-1).ffill().astype(int) #data['states_discretize'].shift(-1).ffill().astype(int)
        data.loc[(data['done_flags']==1) & (data['rewards']==1), 'states_discretize']=alive_state
        data.loc[(data['done_flags']==1) & (data['rewards']==-1), 'states_discretize']=died_state


        data.loc[(data['done_flags']==1) & (data['rewards']==1), 'next_state']=alive_state
        data.loc[(data['done_flags']==1) & (data['rewards']==-1), 'next_state']=died_state

        val =0
        ep = []
        
        for _, valore in enumerate(data['done_flags']):
            ep.append(val)
            if valore ==1:
                val +=1
        
        data['ep']=ep

        data.to_csv(name_file)
        
    patients_id = data.subject_id.unique()

    tmp_idx , test_idx = train_test_split(patients_id, test_size=0.2)
    train_idx, val_idx = train_test_split(tmp_idx, test_size=0.2)

    train_set = data[data['subject_id'].isin(train_idx)]
    val_set = data[data['subject_id'].isin(val_idx)]
    test_set = data[data['subject_id'].isin(test_idx)]
    
    transition_counts = (
        train_set.groupby(['states_discretize', 'actions_discretize', 'next_state'])
        .size()
        .reset_index(name='count')
    )
 
    # probabilità di transizione pr_t = count / somma count per (s,a)
    transition_counts['pr_t'] = (
        transition_counts.groupby(['states_discretize','actions_discretize'])['count']
        .transform(lambda x: x / x.sum())
    )

    # Step 3. reward medio osservato condizionato a (s,a,s')
    mean_rewards = (
        train_set.groupby(['states_discretize','actions_discretize','next_state'])['rewards']
        .mean()
        .reset_index()
    )

    # merge con le probabilità
    pr_r_sans = transition_counts.merge(
        mean_rewards,
        on=['states_discretize','actions_discretize','next_state'],
        how='left'
    )

    # Step 4. reward immediato atteso R(s,a) = somma_{s'} P(s'|s,a) * r(s,a,s')
    pr_r_sans['weighted_reward'] = pr_r_sans['rewards'] * pr_r_sans['pr_t']

    immediate_reward = (
        pr_r_sans.groupby(['states_discretize','actions_discretize'])['weighted_reward']
        .sum()
        .reset_index()
        .rename(columns={'weighted_reward':'immediate_reward'})
    )


    Qn = np.zeros((n_states, n_actions))
    q_peine = peine_mc_iterate(
        train_set[['states_discretize','next_state', 'actions_discretize', 'rewards']].astype(int).to_numpy(),
        Qn, 
        gamma=0.9,
        unsafety_prob=1.0,
        n_epochs=1000,
        learning_rate=0.1,
        safety_map=None
    )

    import scipy

    best_action_indices  = np.nanargmax(q_peine, axis=1)
    action_index_grid = np.tile(np.array(range(7**3)), 662).reshape((662, 7**3))
    best_action_grid = np.repeat(best_action_indices, 7**3).reshape((662, 7**3))
    best_action_bool = best_action_grid == action_index_grid

    mcp_greedy = best_action_bool.astype(float)
    assert (mcp_greedy.sum(axis=1) == 1).all()

    q_mcp_neg = q_peine.copy()[:n_states, :]
    q_mcp_neg[q_mcp_neg == 0.0] = float('-inf')
    mcp_softmax = scipy.special.softmax(q_mcp_neg , axis=1)

    assert mcp_softmax.shape == (n_states, 7**3)
    assert (mcp_greedy.sum(axis=1) == 1).all()

    best_s, best_a = np.unravel_index(np.nanargmax(q_peine), (n_states, 7**3))
    print("Global highest Q value {} for tv, fio2, peep ranges: {}".format(q_peine[best_s, best_a], to_action_ranges(best_a)))
    best_mean_a, best_mean_a_q = np.nanargmax(np.nanmean(q_peine, axis=0)), np.nanmax(np.nanmean(q_peine, axis=0))
    print("Highest avg Q value across states {} for tv, fio2, peep ranges: {}".format(best_mean_a_q, to_action_ranges(best_mean_a)))
    best_med_a, best_med_a_q = np.nanargmax(np.nanmedian(q_peine, axis=0)), np.nanmax(np.nanmedian(q_peine, axis=0))
    print("Highest median Q value across states {} for tv, fio2, peep ranges: {}".format(best_med_a, to_action_ranges(best_med_a)))

    
    def sortnan(x, index):
        return float('-inf') if np.isnan(x[index]) else x[index]
    if plot:
        sns.histplot(q_peine[mcp_greedy == 1.0].ravel(), log_scale=(False, True), bins=200)
        plt.xlabel('Q value')
        plt.title('Histogram of Q values greedy policy')
        plt.show()
        
        ep_idx_succ, ep_idx_fail = create_success_faileure_episode_idx(train_set)
        train_set['positive_outcome'] = train_set['ep'].isin(ep_idx_succ)=='f'

        #estimated_mort_state_visit = train_set.groupby('states_discretize').mean('positive_outcome')[['positive_outcome']].to_numpy()
        estimated_mort_state_visit = train_set.groupby('states_discretize').mean(numeric_only=True)[['positive_outcome']].reindex(range(n_states), fill_value=0).to_numpy()
                
        sns.scatterplot(x=np.nanmean(q_peine, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
        plt.xlabel('Mean estimated Q value')
        plt.ylabel('Average outcome')
        plt.title('Outcome vs mean Q value estimates')
        plt.show()
        sns.scatterplot(x=np.nanmax(q_peine, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
        plt.xlabel('Max estimated Q value')
        plt.ylabel('Average outcome')
        plt.title('Outcome vs max Q value estimates')
        plt.show()
        sns.scatterplot(x=np.nanmedian(q_peine, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
        plt.xlabel('Max estimated Q value')
        plt.ylabel('Average outcome')
        plt.title('Outcome vs median Q value estimates')
        plt.show()
        
        q_vars = np.nanvar(q_peine, axis=1)
        q_means = np.nanmean(q_peine, axis=1)
        q_medians = np.nanmedian(q_peine, axis=1)
        q_maxs = np.nanmax(q_peine, axis=1)
        q_mins = np.nanmin(q_peine, axis=1)
        stacked = np.column_stack((q_means, q_medians, q_maxs, q_mins, q_vars))
        xs = range(n_states)
        means_sorted = np.array(sorted(stacked, key=lambda x: x[0]))
        means_upper = means_sorted[:, 0] + means_sorted[:, -1]
        means_lower = means_sorted[:, 0] - means_sorted[:, -1]
        axs = sns.lineplot(x=xs, y=means_sorted[:, 0])
        axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
        axs.set_ylim(-150, 150)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Mean Q values per state +- 1 var')
        plt.show()

        medians_sorted = np.array(sorted(stacked, key=lambda x: x[1]))
        means_upper = medians_sorted[:, 0] + medians_sorted[:, -1]
        means_lower = medians_sorted[:, 0] - medians_sorted[:, -1]
        axs = sns.lineplot(x=xs, y=medians_sorted[:, 1])
        axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
        axs.set_ylim(-150, 150)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Median Q values per state +- 1 var')
        plt.show()

        mins_sorted = np.array(sorted(stacked, key=lambda x: x[3]))
        axs = sns.scatterplot(x=xs, y=mins_sorted[:, 3], color='orange', alpha=.5)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Min Q values per state')

        maxs_sorted = np.array(sorted(stacked, key=lambda x: x[2]))
        axs = sns.lineplot(x=xs, y=maxs_sorted[:, 2], alpha=.5)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Max Q values per state')
        plt.show()

        q_vars = np.nanvar(q_peine, axis=0)
        q_means = np.nanmean(q_peine, axis=0)
        q_medians = np.nanmedian(q_peine, axis=0)
        q_maxs = np.nanmax(q_peine, axis=0)
        q_mins = np.nanmin(q_peine, axis=0)
        stacked = np.column_stack((q_means, q_medians, q_maxs, q_mins, q_vars))
        xs = range(7**3)
        means_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 0)))
        means_upper = means_sorted[:, 0] + means_sorted[:, -1]
        means_lower = means_sorted[:, 0] - means_sorted[:, -1]
        axs = sns.lineplot(x=xs, y=means_sorted[:, 0])
        axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
        axs.set_ylim(-150, 150)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Mean Q values per action +- 1 var')
        plt.show()

        medians_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 1)))
        means_upper = medians_sorted[:, 0] + medians_sorted[:, -1]
        means_lower = medians_sorted[:, 0] - medians_sorted[:, -1]
        axs = sns.lineplot(x=xs, y=medians_sorted[:, 1])
        axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
        axs.set_ylim(-150, 150)
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Median Q values per action +- 1 var')
        plt.show()

        maxs_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 2)))
        axs = sns.lineplot(x=xs, y=maxs_sorted[:, 2])
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Max Q values per action')
        plt.show()

        mins_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 3)))
        axs = sns.lineplot(x=xs, y=mins_sorted[:, 3])
        plt.xlabel('State')
        plt.ylabel('Q value')
        plt.title('Min Q values per action')
        plt.show()
        
    behavior_policy_df = data.value_counts(['states_discretize', 'actions_discretize']) / data.value_counts(['states_discretize'])
    assert (1.0 - behavior_policy_df.groupby('states_discretize').sum() < 1e10).all(), "Policy action probs should sum to 1 per state"

    behavior_policy_pivot = behavior_policy_df.reset_index().pivot(columns='actions_discretize', index='states_discretize')["count"]
    behavior_policy_pivot = behavior_policy_pivot.reindex(range(n_states), fill_value=0)

    behavior_policy_states = set(behavior_policy_pivot.index.unique())
    for s in range(n_states):
        if s not in behavior_policy_states:
            action_probs = [1.0 / (7**3),] * 7**3 # uniform distribution
            for i, p in enumerate(action_probs):
                behavior_policy_pivot.loc[s] = [s, i, p]

    behavior_policy_pivot = behavior_policy_pivot.sort_values(['states_discretize'])

    for a in range(7**3):
        if a not in behavior_policy_pivot.columns:
            behavior_policy_pivot.loc[:, a] = np.nan

    behavior_policy_nan = behavior_policy_pivot[range(7**3)].to_numpy()
    assert (1- (np.nansum(behavior_policy_nan, axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    behavior_policy = np.nan_to_num(behavior_policy_nan, 0.0)
    assert (1- (behavior_policy.sum(axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    assert behavior_policy.shape == (n_states, 7**3), "Behavior policy should cover all states and actions"
    mcp_greedy_mask = mcp_greedy.astype(bool)
    assert (mcp_greedy_mask.sum(axis=1) == 1).all(), "Greedy policy mask should mask out all-but-one action"

    sns.histplot(scipy.stats.entropy(behavior_policy, axis=1))
    plt.title('Behavior policy per-state entropy')
    plt.xlabel('Entropy') 
    plt.show()

    sns.histplot(scipy.stats.entropy(mcp_softmax, axis=1))
    plt.title('Softmax policy per-state entropy')
    plt.xlabel('Entropy')
    plt.show()

    sns.histplot(behavior_policy[mcp_greedy_mask], log_scale=(False, True))
    plt.xlabel('Action probability greedy policy in behavior policy')
    behavior_policy[mcp_greedy_mask].min(), behavior_policy[mcp_greedy_mask].max()
    plt.show()

    evaluation_policy = mcp_greedy

    behavior_policy_ranks = np.flip(behavior_policy.argsort(axis=1), axis=1)
    ep_bp_ranks = []
    for s in range(n_states):
        ep_a = evaluation_policy[s,:].argmax()
        bp_rank = np.where(behavior_policy_ranks[s, :] == ep_a)[0][0]
        ep_bp_ranks.append(bp_rank)

    sns.histplot(ep_bp_ranks, bins=60)
    plt.title('Greedy policy action ranks in behavior policy')
    plt.xlabel('Rank')
    plt.show()

    behavior_policy_ranked_probs = np.flip(np.sort(behavior_policy, axis=1), axis=1)
    ep_bp_prob_mass = []
    for s in range(n_states):
        ep_a = evaluation_policy[s,:].argmax()
        bp_rank = np.where(behavior_policy_ranks[s, :] == ep_a)[0][0]
        ep_bp_prob_mass.append(behavior_policy_ranked_probs[s, 0:bp_rank].sum())

    sns.histplot(ep_bp_prob_mass)
    plt.title('Probability mass up to greedy actduion')
    plt.xlabel('Action probs')
    plt.show()
    np.array(ep_bp_prob_mass).min(), np.array(ep_bp_prob_mass).max()

