# Classification of Alzheimer's Disease in Mixed EEG Dataset

## Project Overview
This project aims to classify dementia using a mixed EEG dataset containing three classes: AD (Alzheimer's Disease), MCI (Mild Cognitive Impairment), and NC (Normal Control). Multiple datasets were combined and standardized to create a unified dataset for training and evaluating machine learning models.

## Dataset Details
- **Classes and Composition:**
  - **AD:** Contains data from two different datasets.
  - **MCI:** Contains data from a single dataset.
  - **NC:** Contains data from six different datasets.
- **Preprocessing:**
  - Number of electrodes: 19
  - Sampling rate: 250 Hz
  - Segment duration: 30 seconds
  - Applied StandardScaler

## Model and Experiments
Two machine learning models, **Random Forest** and **XGBoost (XGB)**, were employed to perform classification. The experiments were conducted in the following versions:

### Version 1: Initial Multiclass Classification
- NC subsets were **not balanced** or uniformly mixed.
- A randomly selected half of the NC data was used.
- Performed multiclass classification across AD, MCI, and NC.

### Version 2: Balanced Multiclass Classification
- NC subsets were **balanced** by selecting half of the data from each subset.
- Performed multiclass classification across AD, MCI, and NC.

### Version 3: Binary Classification
- Based on the data from **Version 2**.
- Performed binary classification for the following tasks:
  - **AD vs NC**
  - **MCI vs NC**

### Version 4: Binary Classification with CSP Features
- Extended **Version 3** by incorporating **Common Spatial Pattern (CSP)** features.
- Performed binary classification for the following tasks:
  - **AD vs NC**
  - **MCI vs NC**

## Evaluation Metric
- **TBA
