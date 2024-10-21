import os
import re
from itertools import product
from tqdm import tqdm
import numpy as np
import shutil

def get_options(n_classes=0, n_support=0, n_repeats=0, **kwargs):
    return product(range(n_classes * n_support), range(n_repeats))

def get_examples_from_path(n_classes=0, n_support=0, n_repeats=0, path=None, **kwargs):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(path)
    
    options = get_options(n_classes=n_classes, n_support=n_support, n_repeats=n_repeats)
    examples: list[str] = [ [] for _ in range(n_classes * n_support) ]

    for (idx, num), filename in tqdm(zip(options, sorted(os.listdir(path),key=lambda s: tuple(map(int,re.findall(r"\d+",s))))), desc="Loading Existing Augmentations"):

        if not filename.startswith(f"aug-{idx}-{num}"): print(f"error: expected aug-{idx}-{num}..., got {filename}")

        image = os.path.join(path, filename)

        examples[idx].append(image)

    return examples


def get_datasets(paths=[], **kwargs):

    datasets = [ get_examples_from_path(path=path, **kwargs) for path in paths ]
    
    return datasets


def select_example_from_random_dataset(datasets=[], probs=[], idx=0, num=0, paths=[], **kwargs):

    ds_idx = np.random.choice(len(probs), p=probs)

    ds_path = paths[ds_idx]

    image = datasets[ds_idx][idx][num]
    
    return image, ds_path

def get_sample_map(new_path_root=None, **kwargs):

    datasets = get_datasets(**kwargs)
    options = get_options(**kwargs)
    sample_map = []

    for idx, num in options:
        path, ds_path = select_example_from_random_dataset(datasets=datasets, idx=idx, num=num, **kwargs)
        new_path = os.path.join(new_path_root, f"aug-{idx}-{num}.png")
        sample_map.append((path, new_path))

    return sample_map


def copy_files_using_map(copy_map, **kwargs):
    for src, dest in tqdm(copy_map, desc="Copying Images"):
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        shutil.copy(src, dest)
            

def generate_combined_dataset(**kwargs):
    copy_map = get_sample_map(**kwargs)
    copy_files_using_map(copy_map)

def main():
    dataset = "pascal"
    seed = 0
    examples_per_class = 4

    np.random.seed(seed)

    num_classes = 20
    num_synthetic = 10

    new_name = "textual-inversion-0.25-0.5-0.75-1.0-neg-0.5"

    names = [
        "textual-inversion-0.25-neg-0.5",
        "textual-inversion-0.5-neg-0.5",
        "textual-inversion-0.75-neg-0.5",
        "textual-inversion-1.0-neg-0.5",
    ]

    probs = [0.25, 0.25, 0.25, 0.25]

    root_path = "f:/AI dev offload/da-fusion"
    paths = [ f"aug/{name}/{dataset}-{seed}-{examples_per_class}" for name in names ]
    paths = [ os.path.join(root_path, path) for path in paths ]
    new_path = os.path.join(root_path, f"aug/{new_name}/{dataset}-{seed}-{examples_per_class}")

    generate_combined_dataset(
        n_classes=num_classes,
        n_support=examples_per_class,
        n_repeats=num_synthetic,
        paths=paths,
        probs=probs,
        new_path_root=new_path
    )


if __name__ == "__main__":
    main()