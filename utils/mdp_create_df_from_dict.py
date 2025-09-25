import pandas as pd
from d3rlpy.dataset import MDPDataset
from utils.compute_trajectory_discretize import build_trajectories
from config import alive_state, died_state


def build_MDPDataset(df, state_space, action_space, action_size, intermidiate_reward, name= None):
    traj = build_trajectories(df, state_space, action_space, discretize_states=True, name=name, intermidiate_reward=intermidiate_reward)
    #traj.loc[(df['rewards']==100) & (traj['done_flags']==1),'states']= alive_state 
    #traj.loc[(df['rewards']==-100) & (traj['done_flags']==1),'states']= died_state
    
    return MDPDataset(
        traj["states"],
        traj["actions_discretize"],
        traj["rewards"],
        traj["done_flags"],
        action_size=action_size
    )

def create_df_utils(MDP_dataset_dict):
    df = pd.DataFrame({
        "subject_id": MDP_dataset_dict["subject_id"],
        "state": MDP_dataset_dict["states_discretize"],
        "action": MDP_dataset_dict["actions_discretize"],
        "reward": MDP_dataset_dict["rewards"],
        "done": MDP_dataset_dict["done_flags"]
    })
    df['state_action_id'] = df.agg('{0[state]}-{0[action]}'.format, axis=1)
    df['next_state'] = df.state.astype(str).shift(-1)
    
    df.loc[(df.done==1) & (df.reward == -1), 'next_state'] = 650
    df.loc[(df.done==1) & (df.reward == 1), 'next_state'] = 651

    assert df['next_state'].isna().mean()==0, "La correzione non è andata a buon fine"

    df['sans_id'] = df.agg('{0[state_action_id]}-{0[next_state]}'.format, axis=1)

    df['traj_count']=df.groupby('subject_id').cumcount()
    df['traj_count_inv']=df.groupby('subject_id').cumcount(ascending=False)
    df['traj_len'] =df.groupby('subject_id')['traj_count'].transform('count')
    df['traj_count'] +=1 
    df

    pr_state_action = pd.DataFrame(
        (df.state_action_id.value_counts() / df.shape[0])
        ).reset_index().rename(columns={'state_action_id': 'sa_id', 'count': 'pr_sa'})
    pr_state_action

    pr_state_action
    sans_id = '' # state-action-nstate id
    pr_nstate_state_action = pd.DataFrame(df.sans_id.value_counts() / df.shape[0]).reset_index().rename(columns={'index': 'sans_id', 'count': 'pr_sans'})
    pr_nstate_state_action['sa_id'] = list(map(lambda x: '-'.join(x[:2]), pr_nstate_state_action.sans_id.str.split('-')))

    pr_sans = pr_nstate_state_action.merge(pr_state_action, on='sa_id', how='inner').set_index('sans_id')

    # calculate transition probabilities
    pr_trans = pr_sans.pr_sans / pr_sans.pr_sa
    pr_trans = pd.DataFrame(pr_trans).rename(columns={0: 'pr_t'})
    pr_trans

    r_sans = df.groupby('sans_id')['reward'].mean()
    #r_sans.hist()
    #plt.title('Distribution of rewards in s-a-s-r matrix')
    split_r_sans = pd.DataFrame(r_sans.index.str.split('-').tolist()).rename(columns={0:'state', 1: 'action', 2: 'nstate'})
    split_r_sans.state = split_r_sans.state.astype(int)
    split_r_sans.action = split_r_sans.action.astype(int)
    split_r_sans.nstate = split_r_sans.nstate
    r_sans = r_sans.reset_index().merge(split_r_sans, left_index=True, right_index=True)
    r_sans.reward.value_counts()

    pr_r_sans = r_sans.merge(pr_trans, on='sans_id')
    pr_r_sans['weighted_reward'] = pr_r_sans.reward * pr_r_sans.pr_t
    immediate_reward = pr_r_sans.groupby(['state', 'action']).weighted_reward.sum()

    immediate_reward = immediate_reward.reset_index().rename(columns={'state': 'state', 'action': 'action', 'weighted_reward': 'immediate_reward'})
    immediate_reward['state_action_id'] = immediate_reward.agg('{0[state]:.0f}-{0[action]:.0f}'.format, axis=1)

    return df, immediate_reward