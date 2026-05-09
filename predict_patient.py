# predict_patient.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
import random

# COLORS FOR TERMINAL OUTPUT
class Color:
    GREEN  = '\033[92m'
    RED    = '\033[91m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    PURPLE = '\033[95m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'   # resets back to normal color

def print_header():
    print(Color.PURPLE + Color.BOLD + "=" * 40)
    print(" BRAIN TUMOR CLASSIFICATION SYSTEM")
    print(" Gene Expression Based Prediction")
    print("=" * 40)

def print_section(title):
    print(Color.BLUE + Color.BOLD + f"{title}" + Color.RESET)
    print("-" * 50)

# Load saved models
def load_models():
    print_section("\nLoading Models")

    required_files = [
        "model/model.pkl",
        "model/pca.pkl",
        "model/scaler.pkl",
        "model/encoder.pkl",
        "model/feature_names.pkl"
    ]

    # Check all files exist before loading
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(Color.RED + "Missing model files:" + Color.RESET)
        for f in missing:
            print(f"   - {f}")
    
        print()
        print("Please run these scripts first:")
        print(" python preprocess.py")
        print(" python train_model.py")
        sys.exit(1)   # exit the program with error code 1

    model         = pickle.load(open("model/model.pkl",         "rb"))
    pca           = pickle.load(open("model/pca.pkl",           "rb"))
    scaler        = pickle.load(open("model/scaler.pkl",        "rb"))
    encoder       = pickle.load(open("model/encoder.pkl",       "rb"))
    feature_names = pickle.load(open("model/feature_names.pkl", "rb"))

    print(Color.GREEN + " All models loaded successfully" + Color.RESET)
    print(f"   Model type  : Random Forest ({model.n_estimators} trees)")
    print(f"   PCA components: {pca.n_components_}")
    print(f"   Gene features : {len(feature_names)}")
    print(f"   Tumor classes : {list(encoder.classes_)}")

    return model, pca, scaler, encoder, feature_names

# Load patient data
def load_patient_data(feature_names):
    print_section("Load Patient Data")

    print("Choose input method:")
    print("  [1] Enter path to patient CSV file  (real patient data)")
    print("  [2] Use demo data from dataset       (for testing)")
    print()

    # input() pauses the script and waits for the user to type something
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        # Use demo data
        demo_path = "data/Brain_GSE50161.csv"
        if not os.path.exists(demo_path):
            print(Color.RED + f"Demo file not found at: {demo_path}" + Color.RESET)
            sys.exit(1)

        df_full = pd.read_csv(demo_path)

        # Ask how many demo samples
        print()
        n = input("How many demo samples to predict? (1-10, default 5): ").strip()
        n = int(n) if n.isdigit() and 1 <= int(n) <= 10 else 5

        #  Use STRATIFIED random sampling
        all_classes = df_full['type'].unique()
        n_classes   = len(all_classes)

        if n >= n_classes:
            # Pick 1 from each class first (guaranteed variety)
            one_per_class = (
                df_full.groupby('type', group_keys=False).sample(n=1)
                # df_full.groupby('type')
                .apply(lambda x: x.sample(1, random_state=np.random.randint(0, 9999)))
                .reset_index(drop=True)
            )
            # How many more do we need after one-per-class?
            remaining = n - n_classes
            if remaining > 0:
                # Sample remaining randomly from full dataset
                # exclude rows already picked
                already_picked = one_per_class.index
                rest = df_full.drop(index=already_picked, errors='ignore')
                extra = rest.sample(min(remaining, len(rest)),
                                    random_state=np.random.randint(0, 9999))
                sampled_df = pd.concat([one_per_class, extra]).reset_index(drop=True)
            else:
                # n < n_classes: just pick n random rows from one_per_class
                sampled_df = one_per_class.sample(n).reset_index(drop=True)
        else:
            # n is less than number of classes — just pick n random rows
            sampled_df = df_full.sample(n, random_state=np.random.randint(0, 9999)).reset_index(drop=True)

        # Save true labels BEFORE dropping the type column
        true_labels = list(sampled_df['type'].values)

        # Show which classes were selected
        print()
        print("  Randomly selected samples:")
        for i, label in enumerate(true_labels):
            print(f"    Sample {i+1} - true class: {Color.YELLOW}{label}{Color.RESET}")

        # Remove metadata columns for prediction
        df = sampled_df.drop(columns=['samples', 'type'], errors='ignore')

        print()
        print(Color.GREEN + f" Demo data loaded: {n} samples (stratified random)" + Color.RESET)
        return df, true_labels
    else:
        # Real patient data
        print()
        print("Example path: C:\\Users\\Student\\Desktop\\patient_data.csv")
        print("              or just: patient_data.csv  (if in same folder)")
        print()

        file_path = input("Enter CSV file path: ").strip()

        # Remove surrounding quotes if user copied path with quotes
        file_path = file_path.strip('"').strip("'")

        if not os.path.exists(file_path):
            print(Color.RED + f" File not found: {file_path}" + Color.RESET)
            print("Please check the path and try again.")
            sys.exit(1)

        df = pd.read_csv(file_path)
        print(Color.GREEN + f" File loaded: {df.shape[0]} samples, {df.shape[1]} columns" + Color.RESET)

        # Show first few rows
        print()
        print("Preview (first 3 rows, first 5 columns):")
        print(df.iloc[:3, :5].to_string())

        # Remove metadata columns if present
        df = df.drop(columns=['samples', 'type'], errors='ignore')

        true_labels = None  # we don't know true labels for real patients
        return df, true_labels


#Preprocess the patient data
def preprocess(df, scaler, feature_names):
    print_section("Preprocessing Patient Data")

    original_cols = df.shape[1]

    # Remove AFFX control probes
    df = df.loc[:, ~df.columns.str.startswith("AFFX")]
    print(f"   AFFX probes removed : {original_cols - df.shape[1]}")

    # Convert all values to numbers (in case any are stored as text)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill any NaN values with 0
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(Color.YELLOW + f"{nan_count} missing values found — filled with 0" + Color.RESET)
        df = df.fillna(0)

    # Check how many training genes are present in patient data
    patient_cols  = set(df.columns)
    training_cols = set(feature_names)
    overlap       = patient_cols & training_cols
    missing_genes = training_cols - patient_cols

    print(f" Matching genes : {len(overlap)} / {len(feature_names)}")

    if len(missing_genes) > 0:
        print(Color.YELLOW + f" {len(missing_genes)} genes missing — filled with 0" + Color.RESET)
        for gene in missing_genes:
            df[gene] = 0

    if len(overlap) < len(feature_names) * 0.5:
        # Less than 50% match is a serious problem
        print(Color.RED + " Less than 50% gene overlap — data may be from wrong platform" + Color.RESET)
        print("  Expected: Affymetrix HG-U133 Plus 2.0 format")
        sys.exit(1)

    # Reorder columns to exactly match training order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Apply saved scaler — SAME scaling as used during training
    X_scaled = scaler.transform(df)
    print(f" Scaling applied  : StandardScaler (mean=0, std=1)")
    print(Color.GREEN + "Preprocessing complete" + Color.RESET)

    return X_scaled

#Make predictions
def predict(X_scaled, model, pca, encoder):
    print_section("Running Prediction")

    # Apply PCA — reduce dimensions same as training
    X_pca = pca.transform(X_scaled)
    print(f" PCA applied: {X_scaled.shape[1]} genes - {X_pca.shape[1]} components")

    # Get predicted class numbers
    predictions = model.predict(X_pca)

    # Get probability for each class (confidence scores)
    # predict_proba returns array of shape (n_samples, n_classes)
    # Each row sums to 1.0
    probabilities = model.predict_proba(X_pca)

    # Convert numbers back to tumor type names
    tumor_names = encoder.inverse_transform(predictions)

    print(Color.GREEN + f" Predictions complete for {len(tumor_names)} sample(s)" + Color.RESET)

    return tumor_names, probabilities, encoder.classes_

# Display results in a clean, readable format
def display_results(tumor_names, probabilities, class_names, true_labels=None):
    print()
    print(Color.PURPLE + Color.BOLD + "=" * 40)
    print("   PREDICTION RESULTS")
    print("=" * 40 + Color.RESET)

    # Clinical descriptions for each tumor type
    descriptions = {
        'ependymoma':            'Arises from ependymal cells lining ventricles',
        'glioblastoma':          'Most aggressive malignant brain tumor (Grade IV)',
        'medulloblastoma':       'Common malignant pediatric brain tumor',
        'pilocytic_astrocytoma': 'Slow-growing, usually benign (Grade I)',
        'normal':                'Healthy brain tissue — no tumor detected'
    }

    for i, (tumor, probs) in enumerate(zip(tumor_names, probabilities)):
        confidence = probs.max() * 100

        # Choose color based on confidence
        if confidence >= 85:
            conf_color = Color.GREEN
        elif confidence >= 60:
            conf_color = Color.YELLOW
        else:
            conf_color = Color.RED

        print()
        print(Color.BOLD + f"  Patient / Sample {i+1}" + Color.RESET)
        print(f"  {'─' * 45}")

        # Main prediction
        print(f"  Predicted Type  : " +
              Color.BOLD + Color.GREEN + f"{tumor.upper()}" + Color.RESET)

        # True label comparison (only available for demo data)
        if true_labels:
            is_correct = (true_labels[i] == tumor)
            symbol = "✅" if is_correct else "❌"
            label_color = Color.GREEN if is_correct else Color.RED
            print(f"  {symbol} True Label      : " +
                  label_color + f"{true_labels[i]}" + Color.RESET)

        # Confidence
        print(f" Confidence      : " +
              conf_color + f"{confidence:.1f}%" + Color.RESET)

        # Clinical description
        desc = descriptions.get(tumor, "Brain tumor")
        print(f" Description     : {desc}")

        # Probability breakdown for all classes
        print(f"\n  Probability breakdown:")
        # Sort classes by probability (highest first)
        sorted_idx = np.argsort(probs)[::-1]
        for idx in sorted_idx:
            bar_len   = int(probs[idx] * 25)          # scale to 25 chars
            bar       = "█" * bar_len + "░" * (25 - bar_len)
            is_top    = (idx == np.argmax(probs))
            row_color = Color.GREEN if is_top else ""
            print(f"  {row_color}  {class_names[idx]:25s} {bar} {probs[idx]*100:5.1f}%{Color.RESET}")

    print()
    print(Color.PURPLE + "=" * 60 + Color.RESET)


# Save results to CSV
def save_results(tumor_names, probabilities, class_names, true_labels=None):
    print_section("Save Results")

    save = input("Save results to CSV file? (y/n): ").strip().lower()

    if save == 'y':
        # Build results dataframe
        results = []
        for i, (tumor, probs) in enumerate(zip(tumor_names, probabilities)):
            row = {
                'sample_number': i + 1,
                'predicted_type': tumor,
                'confidence_pct': round(probs.max() * 100, 2)
            }
            # Add true label if available
            if true_labels:
                row['true_label'] = true_labels[i]
                row['correct']    = (true_labels[i] == tumor)

            # Add probability for each class
            for cls, prob in zip(class_names, probs):
                row[f'prob_{cls}'] = round(prob, 4)

            results.append(row)

        results_df = pd.DataFrame(results)

        # Default save location
        save_path = "outputs/patient_predictions.csv"
        os.makedirs("outputs", exist_ok=True)

        custom = input(f"Save path (press Enter for default: {save_path}): ").strip()
        if custom:
            save_path = custom.strip('"').strip("'")

        results_df.to_csv(save_path, index=False)
        print(Color.GREEN + f" Results saved to: {save_path}" + Color.RESET)
        print()
        print(results_df.to_string(index=False))
    else:
        print("Results not saved.")

# MAIN runs when you execute  predict_patient.py
if __name__ == "__main__":

    print_header()
    # Run all steps in order
    model, pca, scaler, encoder, feature_names = load_models()
    df, true_labels                            = load_patient_data(feature_names)
    X_scaled                                   = preprocess(df, scaler, feature_names)
    tumor_names, probabilities, class_names    = predict(X_scaled, model, pca, encoder)

    display_results(tumor_names, probabilities, class_names, true_labels)
    # save_results(tumor_names, probabilities, class_names, true_labels)

    print(Color.BOLD + "Prediction session complete.\n" + Color.RESET)