import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

def experiment_plot(result_dirname="pascal-baselines", acc_col="overall_accuracy"):
    agg_metrics = {}

    options_t0 = ["0.25", "0.5", "0.75", "1.0"]
    options_beta = ["0.0", "0.5", "1.0"]

    for beta in options_beta:
        all_accuracies = []

        for t0 in options_t0:
            beta_opt_ext = f"-neg-{beta}" if beta != "0.0" else ""
            results_path = f"{result_dirname}/results-textual-inversion-{t0}{beta_opt_ext}.csv"

            df = pd.read_csv(results_path)
            
            all_accuracies.append(df[acc_col])

        accuracies_mean = [pd.Series(accuracies).mean() for accuracies in all_accuracies]
        accuracies_ci = [pd.Series(accuracies).sem() * 2.364619 for accuracies in all_accuracies]

        agg_metrics[beta] = {
            "mean": accuracies_mean,
            "ci": accuracies_ci,
            "t0": [float(t0) for t0 in options_t0],
        }

    plt.figure(figsize=(6, 4))

    for beta, data in agg_metrics.items():
        plt.errorbar(data["t0"], data["mean"], yerr=data["ci"], label=f'$\\beta={beta}$', marker='o', capsize=5)

    plt.xlabel("Strength $t_0$")
    plt.ylabel("Validation Accuracy")
    #plt.ylim(0.5,0.8)
    plt.legend()
    plt.xticks([0.25,0.5,0.75,1.0])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    

    print("showing overall_accuracy plot")
    experiment_plot(acc_col="overall_accuracy")
    
    # classes = ["Airplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Dining Table", "Dog", "Horse", "Motorcycle", "Person", "Potted Plant", "Sheep", "Sofa", "Train", "Television"]

    # class_acc_cols = [ f"class_accuracy_{cls}" for cls in classes ]
    # for acc_col in class_acc_cols: 
    #     print(f"showing {acc_col} plot")
    #     experiment_plot(acc_col=acc_col)
    
