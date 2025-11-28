"""
Heart Disease Classification Model Training Script
This script loads the heart disease dataset, trains multiple models,
selects the best one, and saves it along with performance metrics.
"""

import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_data(data_path='data/data_cleaned.csv'):
    """Load the heart disease dataset"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Split data into features and target, then train/test sets"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare their performance"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='linear', probability=True)
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_models[name] = model
        print(f"{name} → Accuracy: {acc:.3f}")
    
    # Select best model
    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]
    best_accuracy = results[best_name]
    
    print(f"\n{'='*50}")
    print(f"Best model: {best_name} (Accuracy: {best_accuracy:.3f})")
    print(f"{'='*50}")
    
    return best_model, best_name, results

def save_model_and_metrics(model, model_name, columns, results, y_test, y_pred,
                           model_dir='Model', results_dir='Results'):
    """Save the trained model, columns, and performance metrics"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, f1_score
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    # Save columns
    columns_path = os.path.join(model_dir, 'model_columns.pkl')
    with open(columns_path, 'wb') as f:
        pickle.dump(list(columns), f)
    print(f"Model columns saved to {columns_path}")
    
    # Calculate metrics
    accuracy = results[model_name]
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Save metrics as text file (for CML)
    metrics_txt_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write(f"\nAccuracy = {accuracy:.2f}, F1 Score = {f1:.2f}.")
    print(f"Metrics text saved to {metrics_txt_path}")
    
    # Save performance metrics as JSON
    metrics = {
        'best_model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'all_results': results
    }
    metrics_path = os.path.join(results_dir, 'model_performance.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Performance metrics JSON saved to {metrics_path}")
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plot_path = os.path.join(results_dir, 'model_results.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

def main():
    """Main training pipeline"""
    print("="*50)
    print("Heart Disease Classification Model Training")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, columns = prepare_data(df)
    
    # Train and compare models
    best_model, best_name, results = train_models(X_train, y_train, X_test, y_test)
    
    # Get predictions for confusion matrix
    y_pred = best_model.predict(X_test)
    
    # Save model and metrics
    save_model_and_metrics(best_model, best_name, columns, results, y_test, y_pred)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
