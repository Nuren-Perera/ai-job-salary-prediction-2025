# Predicting AI Job Salaries

A machine learning project built in Python using TensorFlow/Keras and scikit-learn to predict AI job salaries (`salary_usd`) from required skills, education level, job title, and years of experience.

## Overview

The model preprocesses tabular and text features with a `ColumnTransformer` pipeline and trains a feedforward neural network to estimate annual AI salaries in USD. Model performance is evaluated using Mean Squared Error (MSE) and R² score, with visualizations for training/validation loss and actual vs. predicted salaries.

## Repository Contents

| File | Description |
| :--- | :--- |
| `Source Code.py` | Data preprocessing, neural network training, evaluation, and plotting script |
| `ai_job_dataset.csv` | AI job postings dataset (15,000 rows, 19 columns) |
| `Predicting AI Job Salaries.pptx` | Project presentation deck |
| `requirements.txt` | Python package dependencies |

## Dataset & Features

The dataset (`ai_job_dataset.csv`) contains 15,000 job records across 19 attributes. `Source Code.py` uses four input features to predict `salary_usd`:

- **Required Skills (`required_skills`)** — vectorized with `TfidfVectorizer(max_features=1000)`
- **Education Required (`education_required`)** — encoded with `OneHotEncoder`
- **Job Title (`job_title`)** — encoded with `OneHotEncoder`
- **Years of Experience (`years_experience`)** — standardized with `StandardScaler`

## Model Architecture

- **Split:** 80% training / 20% testing (`random_state=42`), with a 10% validation split during training
- **Network:** Keras `Sequential` model with `Dense(128, activation='relu')`, `Dense(64, activation='relu')`, and `Dense(1)` output layer
- **Training:** `adam` optimizer, `mse` loss, `mae` metric, 20 epochs, batch size 32

## Getting Started

1. Clone the repository and navigate into the project folder:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training and evaluation script:
   ```bash
   python "Source Code.py"
   ```
