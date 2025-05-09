import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from data_utils import load_data
from model_utils import nested_cross_validation, train_model, evaluate_model, save_model, load_model
from visualization_utils import visualize_weights, visualize_lambda_selection, visualize_performance_by_lambda

def run_part1(n_components_pca=None, visualize=False):

    N_INNER = 10
    N_OUTER = 10

    base_path = "/Users/askhasle/Documents/GitHub/mathematical-modelling-linear-classification?fbclid=IwZXh0bgNhZW0CMTEAAR4RscCjxDMDP6JY_FFo9OmX-6zw3rEH5aM9SfZn8vUvl7zQZEwZgmj3tH9Pxg_aem__KVEWNgvSE-K9QRLML9l3Q"
    data_path = os.path.join(base_path, "data")
    model_suffix = "_pca" if n_components_pca is not None else ""
    model_save_path = os.path.join(base_path, "models", f"part_1_logistic_regression_model{model_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib")

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    
    print("Loading training data...")
    X_train, y_train = load_data(train_path)
    print("Loading test data...")
    X_test, y_test = load_data(test_path)

    if n_components_pca is not None:
        print(f"Reducing dimensionality using PCA to {n_components_pca} components...")
        pca = PCA(n_components=n_components_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    else:
        print("No PCA applied, using original data dimensions.")

    lambda_values = np.concatenate([
        np.logspace(-5, -3, 5),
        np.linspace(0.001, 0.1, 10), 
        np.linspace(0.1, 1, 5),
        np.linspace(1, 10, 5),
        np.linspace(10, 1000, 5)
    ])
    lambda_values = np.unique(lambda_values.round(8))

    print(f"Running nested cross-validation with {len(lambda_values)} lambda values...")
    best_lambda, best_model, cv_results = nested_cross_validation(X_train, y_train, lambda_values, n_inner=N_INNER, n_outer=N_OUTER)

    # Summarize cross-validation results
    cv_df = pd.DataFrame(cv_results)
    print("\nCross-validation results:")
    print(f"Mean accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"Mean AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    print(f"Selected lambda values across folds: {cv_df['lambda'].tolist()}")
    print(f"Best lambda (most frequently selected): {best_lambda}")

    # Evaluate final model on test set
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print("\nFinal model performance on test set:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")


    print(f"\nSaving the best model trained on all data with lambda={best_lambda}...")
    save_model(best_model, model_save_path)

    if visualize:
        print("Visualizing weights of best model...")
        visualize_weights(best_model)

        print("Visualizing lambda selection frequency...")
        visualize_lambda_selection(cv_df)
        
        print("Visualizing generalization error vs selected lambda...")
        visualize_performance_by_lambda(cv_df, metric='auc')

    # Return results for comparison
    return {
        'best_lambda': best_lambda,
        'cv_df': cv_df,
        'cv_accuracy_list':  cv_df['accuracy'].tolist(),
        'cv_auc_list': cv_df['auc'].tolist(),
        'cv_accuracy_mean': cv_df['accuracy'].mean(),
        'cv_accuracy_std': cv_df['accuracy'].std(),
        'cv_auc_mean': cv_df['auc'].mean(),
        'cv_auc_std': cv_df['auc'].std(),
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
        'y_test': y_test,
        'y_test_pred': test_metrics['y_pred'],
        'y_test_prob': test_metrics['y_prob'],
        'model': best_model
    }

if __name__ == "__main__":
    run_part1(n_components_pca=0.95)