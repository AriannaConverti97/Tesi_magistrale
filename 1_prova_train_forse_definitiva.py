import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import n_actions, state_space, action_space
from utils.mdp_create_df_from_dict import build_MDPDataset
from utils.compute_trajectory_discretize import compute_apache2
from utils.visualize_plot import plot_from_history
from utils.results import evaluate_policy
from Training.utils import select_evaluators, select_model

from d3rlpy.preprocessing import ReturnBasedRewardScaler
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer

INT_REWARD="INT_REWARD"
NO_INT_REWARD="NO_INT_REWARD"
DIR_LOG="d3rlpy_logs"

def main(args):
    data_path = args.data
    seed = args.seed
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    assert (args.device=="cuda" and torch.cuda.is_available()) or True, "GPU not available, using CPU instead"
    device = "cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"

    assert args.model in ["tabular_q", "DQN", "DiscreteBCQ", "DiscreteCQL", "DiscreteSAC", "DiscreteBC", "DiscreteRandom"],  f"Algoritmo non supportato: {args.model}"

    data = pd.read_csv(data_path, keep_default_na=False, na_values=np.nan)
    assert data.notna().all().all(), "The DataFrame contains NaN values."

    if args.intermidiate_reward:
        data['apache2'] = data.apply(compute_apache2, axis=1)
        data['apache2_shifted_up'] = data['apache2'].shift(-1)

    # suddivisione in train, val, test set
    if not os.path.exists(f"Data/train_set_{seed}.h5"):
        patients_ids = data.subject_id.unique()

        tmp_ids, test_ids = train_test_split(patients_ids, test_size=0.2, random_state=seed)
        train_ids, val_ids = train_test_split(tmp_ids, test_size=0.2, random_state=seed)

        splits = {
                "train": data[data.subject_id.isin(train_ids)],
                "val":   data[data.subject_id.isin(val_ids)],
                "test":  data[data.subject_id.isin(test_ids)],
                "train_val": data[data.subject_id.isin(tmp_ids)]
        }

        
        train_set= build_MDPDataset(splits['train'], state_space, action_space, n_actions, args.intermidiate_reward, name="train")
        val_set= build_MDPDataset(splits['val'], state_space, action_space, n_actions,args.intermidiate_reward, name="val")

        test_set= build_MDPDataset(splits['test'], state_space, action_space, n_actions,args.intermidiate_reward, name="test")
        train_val_set= build_MDPDataset(splits['train_val'], state_space, action_space, n_actions, args.intermidiate_reward, name="train_val")

        with open(f"Data/train_val_set_{seed}.h5", "w+b") as f:
            train_val_set.dump(f)
        with open(f"Data/train_set_{seed}.h5", "w+b") as f:
            train_set.dump(f)
        with open(f"Data/val_set_{seed}.h5", "w+b") as f:
            val_set.dump(f)
        with open(f"Data/test_set_{seed}.h5", "w+b") as f:
            test_set.dump(f)  
    else:
        with open(f"Data/train_val_set_{seed}.h5", "rb") as f:
            train_val_set = ReplayBuffer.load(f, InfiniteBuffer())
        with open(f"Data/train_val_set_{seed}.h5", "rb") as f:
            train_set = ReplayBuffer.load(f, InfiniteBuffer())
        with open(f"Data/train_val_set_{seed}.h5", "rb") as f:
            val_set = ReplayBuffer.load(f, InfiniteBuffer())
        with open(f"Data/train_val_set_{seed}.h5", "rb") as f:
            test_set = ReplayBuffer.load(f, InfiniteBuffer())
    
    # TRAINING

    exp_name = f"{args.model}_{INT_REWARD if args.intermidiate_reward else NO_INT_REWARD}_{args.batch_size}_BATCH_SIZE_{args.n_iters}_N_EPOCHS_{seed}_SEED" #_{args.gamma}"
    
    if not os.path.exists(os.path.join(DIR_LOG, exp_name)):
        model = select_model(args.model, device, args.batch_size) #, args.gamma)

        model_history = model.fit(
            train_set,
            n_steps=args.n_iters * len(train_set.episodes) // args.batch_size * args.batch_size ,                 
            n_steps_per_epoch= len(train_set.episodes) // args.batch_size * args.batch_size ,
            experiment_name= exp_name,
            evaluators= select_evaluators("model", val_set),
            with_timestamp=False
        )

        plot_from_history(model_history, select_evaluators("model"), save=True, name=exp_name, show=False)
        # else:
        #     model = load_learnable(os.path.join(DIR_LOG, exp_name))
        
        #if not os.path.exists(os.path.join(DIR_LOG, f"FQE_{exp_name}")):
        fqe = DiscreteFQE(
                    algo=model,
                    config=FQEConfig(
                        batch_size=args.batch_size,
                        reward_scaler=ReturnBasedRewardScaler(return_max=100, return_min=-100)
                    ),
                    device="cuda"
        )

        history_fqe = fqe.fit(train_val_set,
                n_steps= args.n_iters * len(train_val_set.episodes) // args.batch_size * args.batch_size ,
                n_steps_per_epoch=len(train_val_set.episodes) // args.batch_size * args.batch_size ,
                evaluators= select_evaluators("fqe", train_val_set),
                experiment_name= f"Discrete_FQE_{exp_name}",
                with_timestamp=False
            )

        plot_from_history(history_fqe, select_evaluators("fqe"), save=True, name=f"Discrete_FQE_{exp_name}", show=False)
    else:
        print("The model is already trained")
    # #EVALUATIONS

    # initial_states = np.array([ep.observations[0] for ep in train_val_set.episodes])
    
    # q_vals= np.mean(fqe.predict_value(initial_states, model.predict(initial_states)))
    # target_actions =  np.array([model.predict(ep.observations)[0] for ep in train_val_set.episodes])

    # results = evaluate_policy(q_vals, train_val_set, args.model, initial_states, target_actions)
    # #aggiungi la parte delle avg_return come hai fatto in prova_fqe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RL offline - Sacco dataset")
    parser.add_argument("--data", type=str, default="data/dati_imputati.csv")
    parser.add_argument("--model", type=str, required=True, choices=["tabular_q", "DQN", "DiscreteBCQ", "DiscreteCQL", "DiscreteSAC", "DiscreteRandom"],
                        help="Algoritmi da testare")
    parser.add_argument("--n_runs", type=int, default=1, help="Numero di ripetizioni per ogni algoritmo")
    parser.add_argument("--seed", type=int, default=1997)
    #parser.add_argument("--n_steps", type=int, default=100000)
    #parser.add_argument("--n_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128, help="The number of batch_size.")
    #parser.add_argument("--gamma", type=float, default=0.99, help="Gamma value")
    #parser.add_argument("--evaluators", type=str, choices=['all', 'td_error', 'avg_value', 'init_value', 'discount_advantage'])
    parser.add_argument("--n_iters", type=int, default=20, help="Numero di iterazioni/epoche per addestramento")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="cpu o cuda")
    parser.add_argument("--intermidiate_reward", type=bool, default=False, help="Valore intermedio di reward oppure testare con solo valore di ritorno [vivo o morto]")
    args = parser.parse_args()
    main(args)