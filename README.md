# nthu-ml-hw-code

Homework Code of NTHU 機器學習概論 Introduction to Machine Learning (11410CS460200)

## Repository Structure

```
.
├── hw1/                # Lab 1 — used-car price prediction (regression)
├── hw2/                # Lab 2 — patient-outcome classification on messy healthcare data
├── hw3/                # Lab 3 — human activity recognition from wearable-sensor features
├── hw4/                # Lab 6 working copy — diffusion notebook + pretrained MNIST weights
├── hw4_test/           # Lab 4 — product price prediction with a PyTorch NN (+ epoch checkpoints)
├── hw6/                # Lab 6 — conditional diffusion model (DDPM) on MNIST
├── Nvidia workshop/    # NVIDIA DLI "Getting Started with Deep Learning" notebooks + slides
├── pyproject.toml      # Shared uv environment (Python ≥ 3.11)
└── uv.lock
```

## Folder Summaries

### `hw1/` — Lab 1: Used-Car Price Regression
Predict used-car prices from features such as mileage, engine size, fuel efficiency, year, and brand. Contains the main notebook (`Lab1.ipynb`), basic/advanced result CSVs, and train/test data. Set up as its own uv project (`pyproject.toml`, `uv.lock`).

### `hw2/` — Lab 2: Healthcare Outcome Classification
Build classifiers that predict patient outcomes from numeric clinical features (age, smoking history, symptoms, labs), handling missing values along the way. Two notebooks (`Lab2.ipynb`, `Lab2-2.ipynb`) cover the two parts of the lab, with matching output CSVs and train/test data.

### `hw3/` — Lab 3: Human Activity Recognition
Classify human activities from high-dimensional wearable-sensor time-series features. Two notebooks (`Lab3.ipynb`, `Lab3-2.ipynb`) for the two parts, plus part 1/2 output CSVs. The train/test datasets (353MB / 50MB) exceed GitHub's size limits and are git-ignored — download them from the course materials.

### `hw4/` — Lab 6 Working Copy (Diffusion Models)
A working copy of the Lab 6 diffusion-model assignment: `Lab6.ipynb`, the pretrained conditional UNet (`cond_unet_mnist_pretrained.pt`), the public MNIST classifier used for evaluation (`mnist_classifier_public.pt`), and `digits.csv`.

### `hw4_test/` — Lab 4: Product Price Prediction
Predict product prices from technical specs, functional attributes, and brand value using a PyTorch neural network. Includes `Lab4.ipynb`, train/test data, and per-epoch model checkpoints under `model/`.

### `hw6/` — Lab 6: Conditional Diffusion on MNIST
Train/sample a conditional diffusion model (DDPM) that generates handwritten digits. Includes `Lab6.ipynb`, pretrained weights, `digits.csv`, the generated-output folder, and the (git-ignored) auto-downloaded MNIST dataset under `data/`.

### `Nvidia workshop/`
Notebooks and slides from the NVIDIA DLI *Getting Started with Deep Learning* workshop: MNIST basics, ASL image classification with CNNs, data augmentation, transfer learning ("doggy door"), NLP, and the final assessment.

## Environment

The repo root is a [uv](https://docs.astral.sh/uv/) project (Python ≥ 3.11) with the shared ML stack: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, torch, torchvision, tqdm. Run `uv sync` at the root (hw1–hw3 also carry their own standalone uv projects).
