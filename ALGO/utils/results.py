import numpy as np
import matplotlib.pyplot as plt

def evaluate_policy(q_vals, dataset, model_name, initial_states, target_actions ):
    results={}

    n_trajectories = len(initial_states)
    avg_return_fqe = np.mean(q_vals)
    
    boot_means = []
    for _ in range(1000):
        idx = np.random.choice(n_trajectories, n_trajectories, replace=True)
        boot_means.append(np.mean(q_vals[idx]))
    ci_lower, ci_upper = np.percentile(boot_means, [5,95])

    sum_rewards = np.array([np.sum(ep.rewards) for ep in dataset.episodes])
    avg_return_obs = np.mean(sum_rewards)
    variance = np.var(sum_rewards)
    success_rate = np.mean([1 if np.sum(ep.rewards) >= 60 else 0 for ep in dataset.episodes])
    violation_rate = np.mean([1 if min(ep.rewards) < -60 else 0 for ep in dataset.episodes])

    behavior_actions = np.array([ep.actions[0] for ep in dataset.episodes])
    action_deviation = np.mean([1 if a != b else 0 for a,b in zip(target_actions, behavior_actions)])

    results[model_name] = {
        "avg_return_fqe": avg_return_fqe,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "avg_return_obs": avg_return_obs,
        "variance": variance,
        "success_rate": success_rate,
        "violation_rate": violation_rate,
        "action_deviation": action_deviation
    }
   
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Avg expected return (FQE): {res['avg_return_fqe']:.3f} [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
        print(f"  Avg return observed: {res['avg_return_obs']:.3f}")
        print(f"  Variance: {res['variance']:.3f}")
        print(f"  Success rate: {res['success_rate']:.2f}")
        print(f"  Violation rate: {res['violation_rate']:.2f}")
        print(f"  Action deviation: {res['action_deviation']:.2f}")

    labels = list(results.keys())
    avg_fqe = [results[name]["avg_return_fqe"] for name in labels]
    ci_lowers = [results[name]["ci_lower"] for name in labels]
    ci_uppers = [results[name]["ci_upper"] for name in labels]
    action_dev = [results[name]["action_deviation"] for name in labels]

    plt.figure(figsize=(10,6))
    plt.bar(labels, avg_fqe, yerr=[np.array(avg_fqe)-np.array(ci_lowers), np.array(ci_uppers)-np.array(avg_fqe)],
            capsize=5, color='skyblue', label="Avg expected return (FQE)")
    plt.ylabel("Avg expected return")
    plt.title("Confronto policy target con intervallo di confidenza")
    plt.twinx()
    plt.plot(labels, action_dev, color='red', marker='o', label="Action deviation")
    plt.ylabel("Action deviation")
    plt.legend(loc='upper right')
    plt.show()

    return results