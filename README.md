# Brain Tumor Classification Using Gene Expression Data

A machine learning project that predicts brain tumor type from gene expression microarray data.
Built as Minor Project-I (BTCSE 601) at Jamia Hamdard, New Delhi.


## What This Project Does

This project takes gene expression data from brain tissue samples and predicts which type of brain tumor the patient has. The dataset has 130 patient samples, each with 54,613 gene expression values. Because there are far more features than samples, we first reduce dimensions using PCA, then classify using Random Forest.

The pipeline runs in four Python scripts in order:

1. explore_data.py — loads the dataset and prints basic information
2. preprocess.py — cleans and prepares the data for machine learning
3. train_model.py — applies PCA, trains the model, and evaluates it
4. predict_patient.py — takes a new patient CSV and predicts the tumor type


## Dataset

The dataset is GSE50161, available freely from NCBI Gene Expression Omnibus (GEO).

It contains 130 samples across five categories:

- Ependymoma: 46 samples
- Glioblastoma: 34 samples
- Medulloblastoma: 22 samples
- Pilocytic Astrocytoma: 15 samples
- Normal brain tissue: 13 samples

The data was measured using the Affymetrix HG-U133 Plus 2.0 microarray chip and pre-processed using RMA normalization before release. Values are in log2 scale and there are no missing values.

Dataset link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50161


## Requirements

Python 3.8 or above is required. Install all dependencies with:

    pip install -r requirements.txt

The requirements file contains:

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn


## How to Run

First place the dataset CSV file inside a folder called data:

    data/Brain_GSE50161.csv

Then run the four scripts in order:

    python explore_data.py
    python preprocess.py
    python train_model.py
    python predict_patient.py

Each script saves its output automatically. You do not need to pass any arguments.


## Script 1 — explore_data.py

This script loads the raw dataset and prints the following information without modifying anything:

- Total number of rows and columns
- First few rows to see the data structure
- Count of samples per tumor type and class imbalance ratio
- Whether any values are missing
- Minimum, maximum, and mean gene expression values

It also saves a bar chart of the class distribution to the outputs folder.

---

## Script 2 — preprocess.py

This script prepares the raw data for machine learning. The steps are performed in a strict order because the train-test split must happen before scaling to prevent data leakage.

Separating features and labels: The tumor type column is saved separately before anything is removed. Gene expression columns become the input features called X and tumor type becomes the label called y.

Removing AFFX control probes: The Affymetrix chip includes 62 technical quality control probes whose names start with AFFX. These are not human genes and are removed to avoid adding noise. This leaves 54,613 gene columns.

Label encoding: The five tumor type names are converted to numbers because machine learning models only work with numbers. Ependymoma becomes 0, glioblastoma 1, medulloblastoma 2, normal 3, and pilocytic astrocytoma 4. The encoder is saved so predictions can be converted back to names later.

Train-test split: The data is split into 80 percent training (104 samples) and 20 percent testing (26 samples). The split uses stratify so that each tumor type appears proportionally in both sets. This matters because the normal class only has 13 samples and could easily end up entirely in training without stratification. A fixed random seed makes the split reproducible.

Feature scaling: StandardScaler transforms each gene column so that across the training set, every gene has mean equal to zero and standard deviation equal to one. This is necessary because PCA is sensitive to the scale of values. The key rule is that the scaler learns from training data only and the same learned parameters are applied to test data without any new learning. Fitting the scaler on test data would allow test information to influence training, which is called data leakage.

All processed arrays and Python objects are saved to files so the next scripts can load them directly without repeating any work.


## Script 3 — train_model.py

This script reduces dimensions, trains the model, and evaluates it.

Finding the right number of PCA components: PCA is first run with all possible components on training data. We look at how much variance each additional component explains. The cumulative variance curve shows that 50 components retain approximately 88 percent of total variance. This is a good balance between keeping information and keeping the model fast. The memory-heavy full PCA object is deleted after we find this number.

Applying PCA: A new PCA with 50 components is fitted on training data only, then the same transformation is applied to both training and test data. X_train reduces from 104 samples by 54,613 genes to 104 samples by 50 components. X_test reduces similarly.

Training Random Forest: A Random Forest of 200 decision trees is trained on the 50 PCA components. Each tree is built on a random subset of samples and features, which makes the ensemble more reliable than any single tree. The max depth is set to 10 to prevent trees from memorizing training data. The class weight is set to balanced so that rare classes like normal (13 samples) are automatically given more importance during training.

Cross validation: The model is tested using stratified 5-fold cross validation. The training data is divided into 5 folds and the model trains on 4 folds then validates on 1, rotating through all 5 combinations. The macro F1 score is computed for each fold. The mean and standard deviation across 5 folds give a more reliable estimate of real performance than a single split.

Final evaluation: The model is evaluated once on the 26 test samples it has never seen. Output includes overall accuracy, and per-class precision, recall, and F1-score. A confusion matrix heatmap is saved showing exactly which classes were confused with which.

The trained model and fitted PCA are saved as pickle files for use by the prediction script.


## Script 4 — predict_patient.py

This script lets you classify new patient samples from the terminal without needing a web browser.

When you run it, you choose between two modes. Mode 1 is for real patient data. You type the path to a CSV file. The file should have one row per patient and one column per gene probe, with log2 expression values. Mode 2 is for testing with demo data. It randomly picks samples from the original dataset, guaranteeing at least one sample from each tumor class.

After loading, the script applies the exact same preprocessing steps that were used during training: removing AFFX probes, converting all values to numbers, filling any missing gene columns with zero, reordering all columns to exactly match the training order, and scaling using the saved scaler parameters.

The processed data then passes through the saved PCA transformer and into the Random Forest. Results are printed showing the predicted tumor type, confidence percentage, clinical description, and a breakdown of probabilities for all five classes. In demo mode the true label is also shown. At the end you can choose to save results as a CSV file.


## Results

The model achieved a test accuracy of 88.46 percent, meaning 23 out of 26 held-out test samples were correctly classified.

Glioblastoma was classified perfectly with 7 out of 7 correct. Normal tissue was also classified perfectly with 3 out of 3 correct. Ependymoma had 8 out of 9 correct with one misclassified as pilocytic astrocytoma. Pilocytic astrocytoma had 3 out of 3 correct. Medulloblastoma was the hardest class with only 2 out of 4 correct. The other 2 were misclassified as ependymoma because these tumor types have overlapping gene expression profiles.


## Folder Structure

    BrainTumorProject/
    |
    |-- data/
    |   |-- Brain_GSE50161.csv
    |   |-- processed/
    |       |-- X_train.npy
    |       |-- X_test.npy
    |       |-- y_train.npy
    |       |-- y_test.npy
    |
    |-- model/
    |   |-- model.pkl
    |   |-- pca.pkl
    |   |-- scaler.pkl
    |   |-- encoder.pkl
    |   |-- feature_names.pkl
    |
    |-- outputs/
    |   |-- class_distribution.png
    |   |-- pca_variance_curve.png
    |   |-- pca_2d_scatter.png
    |   |-- confusion_matrix.png
    |   |-- patient_predictions.csv
    |
    |-- explore_data.py
    |-- preprocess.py
    |-- train_model.py
    |-- predict_patient.py
    |-- requirements.txt
    |-- README.md


## Why These Decisions Were Made

PCA was chosen because with 54,613 gene features and only 130 samples, any model trained directly would memorize noise instead of learning real patterns. PCA compresses the data to 50 components while keeping 88 percent of the information.

Random Forest was chosen because it builds 200 trees each trained on a random subset, and the majority vote across all trees is more reliable than any single decision. It also works well with the balanced class weight setting for handling imbalanced data.

The train-test split happens before scaling because if you scale the full dataset first, the test data statistics influence the scaler. This makes the model appear better than it really is. Always split first, then fit the scaler on training data only.

Stratification during the split ensures that rare classes like normal (13 samples) appear in both training and test sets in proportion, rather than ending up entirely in one set by random chance.


## Limitations

The dataset has only 130 samples which limits how well the model generalizes to new patients from different hospitals or populations. All evaluation is done on the same dataset and the model has not been tested on independent external data. The prediction tool only works with Affymetrix HG-U133 Plus 2.0 microarray data. Medulloblastoma had low F1 because only 4 test samples were available and the class has biological overlap with ependymoma.

## Possible Future Extensions

Testing the model on a second brain tumor dataset from NCBI GEO such as GSE4290 would show whether it truly generalizes. Comparing with SVM or XGBoost would show whether Random Forest is the best choice for this data. Adding biomarker discovery using ANOVA could identify which specific genes most strongly distinguish each tumor type, which is useful for clinical research. A simple web interface using Streamlit would let users upload files and get predictions without using the command line.

