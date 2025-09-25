from d3rlpy.metrics.evaluators import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator, \
        AverageValueEstimationEvaluator, InitialStateValueEstimationEvaluator, \
        DiscreteActionMatchEvaluator, SoftOPCEvaluator
from d3rlpy.preprocessing import ReturnBasedRewardScaler

def select_model(model_name, device, batch_size): #gamma
    """
    Selects and initializes a discrete RL model from d3rlpy.

    Parameters:
    - model_name (str): Name of the model ("DiscreteBCQ", "DiscreteCQL", "DiscreteSAC", "DQN").
    - device (str): Device to run the model on, e.g., "cpu" or "cuda:0".
    - batch_size (int): Batch size for training.

    Returns:
    - An instance of the requested d3rlpy model initialized with a reward scaler.
    
    Raises:
    - ValueError: if the model_name is not supported.
    """
    reward_scaler = ReturnBasedRewardScaler(return_max=100, return_min=-100)
     
    models = {
        "DiscreteBCQ": lambda: __import__('d3rlpy.algos', fromlist=['DiscreteBCQ']).DiscreteBCQ(
            __import__('d3rlpy.algos', fromlist=['DiscreteBCQConfig']).DiscreteBCQConfig(
                batch_size=batch_size,
                #gamma=gamma,
                reward_scaler=reward_scaler
            ),
            enable_ddp=False,
            device=device),            
 
        "DiscreteCQL":lambda: __import__('d3rlpy.algos', fromlist=['DiscreteCQL']).DiscreteCQL(
            __import__('d3rlpy.algos', fromlist=['DiscreteCQLConfig']).DiscreteCQLConfig(
                epoch_size=batch_size,
                #gamma=gamma,
                reward_scaler=reward_scaler
            ),
            enable_ddp=False,
            device=device),  

        "DiscreteSAC":lambda: __import__('d3rlpy.algos', fromlist=['DiscreteSAC']).DiscreteSAC(
            __import__('d3rlpy.algos', fromlist=['DiscreteSACConfig']).DiscreteSACConfig(
                reward_scaler=reward_scaler
            ),
            enable_ddp=False,
            device=device),  

        "DQN":lambda: __import__('d3rlpy.algos', fromlist=['DQN']).DQN(
            __import__('d3rlpy.algos', fromlist=['DQNConfig']).DQNConfig(
                reward_scaler=reward_scaler
                ),
            enable_ddp=False,
            device=device)
        # "DiscreteRandom":lambda: __import__('d3rlpy.algos', fromlist=['DiscreteRandomPolicy']).DiscreteRandomPolicy(
        #     __import__('d3rlpy.algos', fromlist=['DiscreteRandomPolicyConfig']).DiscreteRandomPolicyConfig(
        #         reward_scaler=ReturnBasedRewardScaler(return_max=100, return_min=-100),
        #     )),  
    }

    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return models[model_name]()

def select_evaluators(name, dataset=None):
    """
    Returns evaluators for a model or FQE.

    If dataset is None, returns only the names of the evaluators.
    Otherwise, returns the corresponding evaluator objects initialized
    with the episodes from dataset.
    
    Parameters:
    - name (str): either "model" or "fqe" to select the type of evaluators.
    - dataset (optional): dataset containing episodes; if None, only names are returned.

    Returns:
    - list of str or dict: evaluator names if dataset is None, 
      otherwise a dictionary mapping names to evaluator objects.
    """
    if name == "model":
        if dataset is None:
            return ['td_error', 'avg_value', 'init_value', 'discount_advantage', 'discrete_action_match']
        return {
            "td_error": TDErrorEvaluator(dataset.episodes),
            "avg_value": AverageValueEstimationEvaluator(dataset.episodes),
            "init_value": InitialStateValueEstimationEvaluator(dataset.episodes),
            "discount_advantage": DiscountedSumOfAdvantageEvaluator(dataset.episodes),
            "discrete_action_match": DiscreteActionMatchEvaluator(dataset.episodes)
        }
    
    elif name == "fqe":
        if dataset is None:
            return ['init_value', 'soft_OPC']
        return {
            "init_value": InitialStateValueEstimationEvaluator(dataset.episodes),
            "avg_value": AverageValueEstimationEvaluator(dataset.episodes),
            "soft_OPC": SoftOPCEvaluator(100, dataset.episodes)
        }
    
    else:
        raise ValueError(f"Unknown evaluator type: {name}")