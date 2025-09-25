import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

INT_REWARD="INT_REWARD"
NO_INT_REWARD="NO_INT_REWARD"
PATH = "img"
PATH_LOSS = "img\loss"

def plot_from_history(history, metrics, show=True, save=False, name=None):
    
    epochs = [x[0] for x in history]
    
    if 'soft_OPC' in metrics:
        losses=['loss']
        n_rows, n_cols= 1, 3
        figsize=(20,8)
    else:
        losses = ['td_loss', 'imitator_loss', 'loss']
        n_rows, n_cols = 2, 3
        figsize=(20,10)
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, m in enumerate(metrics):
        value = [x[1][m] for x in history]
        axes[i].plot(epochs, value, label=m, color='tab:blue')
        axes[i].set_title(f"{m} Training")
        axes[i].set_ylabel(m)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()

    for l in losses:
        value = [x[1][l] for x in history]
        axes[-1].plot(epochs, value, label=l)

    axes[-1].set_title(f"Loss Training")
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_ylabel(f"Loss")
    axes[-1].legend()
    axes[-1].grid(True)


    if save:
        name_file_to_save = os.path.join(PATH, f"all_metrics_loss_{name}.png")
        plt.savefig(name_file_to_save, dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()  
    
def find_file_loss(path):
    file_names=[]
    names=[]
    for file in os.listdir(path):
        if "loss" in file:
            names.append(file.replace(".csv", ""))
            file_names.append(os.path.join(path,file))
    return file_names, names

def plot_loss(paths, save=False):
    for path in paths:
        tmp = path.replace("d3rlpy_logs/","")
        model_name= tmp.split("_")[0]
       
        file_names, names= find_file_loss(path)
        for num, file in enumerate(file_names):
            data = pd.read_csv(file, header=None, names=["0", "timesteps", "value"])
            plt.plot(data['timesteps'], data['value'], label=names[num])
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Training History')
            plt.legend()

        if save:
            name_file_to_save= f"img\loss\{tmp}_loss.png"
            plt.savefig(name_file_to_save, dpi=300)

        plt.show()
    
def plot(paths, metrics, intermidiate=False, all_in_one=False, save=False, given_name=None):
    select_paths = [p for p in paths if "BEHAVIOR" not in p]
    if intermidiate:
        usage_paths = [p for p in select_paths if "NO_INT" not in p]
    else:
        usage_paths= [p for p in select_paths if "NO_INT" in p]
    
    if all_in_one:
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(20,10))
        axes = axes.flatten() 

        for i, metric in enumerate(metrics):
            ax = axes[i]
            name_metric = metric.replace("_", " ").capitalize()
            for path in usage_paths:
                name_algo = os.path.basename(path).split('_')[0]
                file_name = os.path.join(path, f"{metric}.csv")
                assert os.path.exists(file_name), f"File not found: {file_name}"
                data = pd.read_csv(file_name, header=None, names=["0", "timesteps", "value"])
                ax.plot(data['timesteps'], data['value'], label=name_algo)

            ax.set_title(name_metric)
            ax.grid(True)
            ax.legend()

        # Nascondiamo eventuali subplot vuoti
        for j in range(len(metrics), rows*cols):
            axes[j].axis('off')

        plt.tight_layout()
        if save:
            file_name_to_save = f"img/all_metrics_{INT_REWARD if intermidiate else NO_INT_REWARD}.png" if given_name is None else f"img/{given_name}.png"
            plt.savefig(file_name_to_save, dpi=300)
        plt.show()

    else:
        for metric in metrics:
            name_metric = metric.replace("_", " ").capitalize()
            for path in usage_paths:
                name_algo = os.path.basename(path).split('_')[0]
                file_name= os.path.join(path, f"{metric}.csv")
                assert os.path.exists(file_name), f"File not found: {file_name}"
                data = pd.read_csv(file_name, header=None, names=["0", "timesteps", "value"])
                #assert data.notna().all()
                plt.plot(data['timesteps'], data['value'], label=name_algo)
        
            plt.title(f"{name_metric}")
            plt.grid(True)
            plt.legend()
            if save:
                name_file_to_save= f"img\{metric}_{INT_REWARD if intermidiate else NO_INT_REWARD}.png"
                plt.savefig(name_file_to_save, dpi=300)
            plt.show()

if __name__ == "__main__":
    directory = "d3rlpy_logs/"
    paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    metrics=['td_error', 'avg_value', 'init_value', 'discount_advantage', 'discrete_action_match']
    #plot_loss(paths)

    #plot(['d3rlpy_logs\DiscreteBCQ_NO_INT_REWARD_20250916005841', 'd3rlpy_logs\DiscreteCQL_NO_INT_REWARD_20250916033107'], metrics, all_in_one=True, save=True, given_name=f"all_metrics_NO_INT_REW__BCQ_CQL")
    #plot(paths, metrics, all_in_one=True, save=True)
    #plot(paths, metrics, intermidiate=True, all_in_one=True , save=True)
    plot(['d3rlpy_logs\DiscreteBCQ_NO_INT_REWARD_128_BATCH_SIZE_20_N_EPOCHS_20250925003334'], metrics=metrics, all_in_one=True)
    plot_loss(['d3rlpy_logs\DiscreteBCQ_NO_INT_REWARD_128_BATCH_SIZE_20_N_EPOCHS_20250925003334'])
