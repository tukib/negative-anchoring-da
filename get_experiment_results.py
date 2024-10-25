import os
import pandas as pd

def get_accuracies(df: pd.DataFrame):

    accuracies = df[df["metric"] == "Accuracy"]["value"]

    return accuracies


def get_class_accuracies(df: pd.DataFrame):

    class_accuracies = df[df["metric"].str.contains("Accuracy") & (df["metric"] != "Accuracy")]

    return class_accuracies


def get_last_epoch(df: pd.DataFrame):

    last_epoch = df["epoch"].max()

    return last_epoch


def get_best_epoch(df: pd.DataFrame):
    
    accuracies = get_accuracies(df)

    best_epoch = df.loc[accuracies.idxmax(), "epoch"]

    return best_epoch


def get_epoch_for_results(df: pd.DataFrame, last_epoch: bool=None):

    if last_epoch: 
        epoch = get_last_epoch(df)
    else:
        epoch = get_best_epoch(df)

    return epoch


def metrics_dict_from_epoch(df: pd.DataFrame, epoch: int):

    epoch_df = df[df["epoch"] == epoch]

    overall_accuracy = get_accuracies(epoch_df).values[0] # already filtered epoch, take single accuracy

    metrics_dict = {
        "epoch": epoch,
        "overall_accuracy": overall_accuracy,
    }

    class_accuracies = get_class_accuracies(epoch_df)
    for _, row in class_accuracies.iterrows():

        class_name = row["metric"].replace("Accuracy ", "")
        metrics_dict[f"class_accuracy_{class_name}"] = row["value"]

    return metrics_dict


def compile_experiment_results(strategy,
                               iterations: int=None,
                               examples_per_class: int=None,
                               last_epoch: bool=None,
                               result_dirname: str=None,
                               save_csv: bool=None,
                               save_plot: bool=None):

    metrics = []

    for i in range(iterations):
        file_path = f"{result_dirname}/{strategy}/results_{i}_{examples_per_class}.csv"
        df = pd.read_csv(file_path)

        validation_df = df[df["split"] == "Validation"]

        # epoch = get_epoch_for_results(validation_df, last_epoch=last_epoch) # prior work used val?
        epoch = get_epoch_for_results(df[df["split"] == "Training"], last_epoch=last_epoch)

        metrics_dict = { "i": i, **metrics_dict_from_epoch(validation_df, epoch) }
        metrics.append(metrics_dict)

    metrics_df = pd.DataFrame(metrics)

    if save_csv:
        output_path = f"{result_dirname}/results-{strategy}.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"Saved compiled results to {output_path}")

    if save_plot: create_plot(metrics_df)


def create_plot(df: pd.DataFrame):

    pass


def main(
        result_dirname="pascal-baselines",
        strategies=[
            # 0.25
            "textual-inversion-0.25",
            "textual-inversion-0.25-neg-0.5",
            "textual-inversion-0.25-neg-1.0",
            # 0.5
            "textual-inversion-0.5",
            "textual-inversion-0.5-neg-0.5",
            "textual-inversion-0.5-neg-1.0",
            # 0.75
            "textual-inversion-0.75",
            "textual-inversion-0.75-neg-0.5",
            "textual-inversion-0.75-neg-1.0",
            # 1.0
            "textual-inversion-1.0",
            "textual-inversion-1.0-neg-0.5",
            "textual-inversion-1.0-neg-1.0",
        ],
        iterations_list=[
            # 0.25
            1, 1, 1,
            # 0.5 
            1, 1, 1,
            # 0.75
            8, 8, 8,
            # 1.0
            1, 1, 1,
        ],
        examples_per_class_list=[4 for _ in range(3*4)],
        last_epoch=True,
        save_csv=True,
        save_plot=True,
):
    
    for strategy, iterations, examples_per_class in zip(strategies,
                                                        iterations_list,
                                                        examples_per_class_list):
        
        compile_experiment_results(strategy,
                                   iterations=iterations,
                                   examples_per_class=examples_per_class,
                                   last_epoch=last_epoch,
                                   result_dirname=result_dirname,
                                   save_csv=save_csv,
                                   save_plot=save_plot)


if __name__ == "__main__":
    main()
