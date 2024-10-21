from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch
import numpy as np
import abc
import random
import os
import re

class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 negative_probability: float = 0.0):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.negative_probability = negative_probability

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        if self.synthetic_dir is not None and os.path.exists(self.synthetic_dir) and len(os.listdir(self.synthetic_dir)) > 0:

            print("Loading existing dataset. PRNG will diverge from direct augment->train, so ensure evaluation does not use both direct and deferred datasets for reproducibility.")

            for (idx, num), filename in tqdm(zip(options, sorted(os.listdir(self.synthetic_dir),key=lambda s: tuple(map(int,re.findall(r"\d+",s))))), desc="Loading Existing Augmentations"):

                if not filename.startswith(f"aug-{idx}-{num}"): print(f"error: expected aug-{idx}-{num}..., got {filename}")

                image = os.path.join(self.synthetic_dir, filename)
                label = self.get_label_by_idx(idx)

                self.synthetic_examples[idx].append((image, label))

            return

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
            metadata = self.get_metadata_by_idx(idx)

            use_negative_example = np.random.uniform() < self.negative_probability
            aug_base_idx = idx
            if use_negative_example:
                while label == self.get_label_by_idx(aug_base_idx):
                    aug_base_idx = np.random.randint(len(self))
                image = self.get_image_by_idx(aug_base_idx)

            image, label = self.generative_aug(
                image, label, metadata)

            if self.synthetic_dir is not None:

                filename = f"aug-{idx}-{num}-neg{aug_base_idx}.png" if use_negative_example else f"aug-{idx}-{num}.png"
                pil_image, image = image, os.path.join(
                    self.synthetic_dir, filename)

                pil_image.save(image)

            self.synthetic_examples[idx].append((image, label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label