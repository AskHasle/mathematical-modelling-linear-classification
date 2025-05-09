import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

def visualize_weights(model, image_shape=(224, 224)):
    weights = model.coef_[0]
    
    weight_image = weights.reshape(image_shape)
    abs_weight_image = np.abs(weight_image)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original weights
    im0 = axes[0].imshow(weight_image, cmap='seismic')
    axes[0].set_title('Original Model Weights\n(Blue = Negative, Red = Positive)')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot absolute weights
    im1 = axes[1].imshow(abs_weight_image, cmap='viridis')
    axes[1].set_title('Absolute Model Weights')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def visualize_lambda_selection(cv_df):
    if 'lambda' not in cv_df.columns:
        raise ValueError("cv_df must contain a 'lambda' column.")

    plt.figure(figsize=(10, 6))
    lambda_counts = cv_df['lambda'].value_counts().sort_index()
    lambda_labels = lambda_counts.index.astype(str)
    counts = lambda_counts.values

    plt.bar(lambda_labels, counts)
    plt.xlabel('Lambda Value')
    plt.ylabel('Selection Frequency')
    plt.title('Lambda Selection Frequency in Cross-Validation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_performance_by_lambda(cv_df, metric):
    if metric not in cv_df.columns:
        print(f"Warning: Metric '{metric}' not found in cv_results. Available metrics: {cv_df.columns.tolist()}")
        return

    performance_by_lambda = cv_df.groupby('lambda')[metric].agg(['mean', 'std']).reset_index()
    performance_by_lambda = performance_by_lambda.sort_values('lambda') # Ensure sorted for plotting

    plt.figure(figsize=(10, 6))
    plt.errorbar(performance_by_lambda['lambda'], performance_by_lambda['mean'],
                 yerr=performance_by_lambda['std'], fmt='-o', capsize=5,
                 label=f'Mean Outer Fold {metric.upper()} ± Std Dev')

    plt.xscale('log')
    plt.xlabel('Selected Lambda (log scale)')
    plt.ylabel(f'Outer Fold {metric.upper()}')
    plt.title(f'Generalization Performance vs. Selected Lambda ({metric.upper()})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_pca_2d(X, y, title='PCA of X-ray Data (2 Components)'):
    if X.shape[0] == 0:
        print("No data provided to plot_pca_2d.")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    X_pca_positive = X_pca[y == 1]
    X_pca_negative = X_pca[y == 0]

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca_negative[:, 0], X_pca_negative[:, 1], alpha=0.7, label='Healthy (0)', c='green')
    plt.scatter(X_pca_positive[:, 0], X_pca_positive[:, 1], alpha=0.7, label='Pneumonia (1)', c='red')

    plt.title(title)
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Explained variance ratio by PC1 and PC2: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by PC1 and PC2: {np.sum(pca.explained_variance_ratio_)}")

def plot_pca_3d(X, y, title='PCA of X-ray Data (3 Components)'):
    if X.shape[0] == 0:
        print("No data provided to plot_pca_3d.")
        return

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    X_pca_positive = X_pca[y == 1]
    X_pca_negative = X_pca[y == 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_pca_negative[:, 0], X_pca_negative[:, 1], X_pca_negative[:, 2], alpha=0.7, label='Healthy (0)', c='green')
    ax.scatter(X_pca_positive[:, 0], X_pca_positive[:, 1], X_pca_positive[:, 2], alpha=0.7, label='Pneumonia (1)', c='red')

    ax.set_title(title)
    ax.set_xlabel('Principal Component 1 (PC1)')
    ax.set_ylabel('Principal Component 2 (PC2)')
    ax.set_zlabel('Principal Component 3 (PC3)')
    ax.legend()
    plt.grid(True)
    plt.show()

    print(f"Explained variance ratio by PC1, PC2, and PC3: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by PC1, PC2, and PC3: {np.sum(pca.explained_variance_ratio_)}")

def plot_pca_scree(X, title='PCA Scree Plot'):
    if X.shape[0] == 0:
        print("No data provided to plot_pca_scree.")
        return

    num_total_features = X.shape[1]
    if num_total_features == 0:
        print("Input data X has no features to perform PCA on.")
        return

    # Determine the number of components to compute and display (max 8)
    n_components_to_analyze = min(8, num_total_features)

    # Perform PCA for the specified number of components
    pca = PCA(n_components=n_components_to_analyze)
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # n_components here refers to the number of components we are displaying
    n_components_displayed = len(explained_variance_ratio) 

    plt.figure(figsize=(10, 6))

    # Plot explained variance for each component
    plt.bar(range(1, n_components_displayed + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual explained variance')

    # Plot cumulative explained variance
    plt.step(range(1, n_components_displayed + 1), cumulative_explained_variance, where='mid',
            label='Cumulative explained variance', color='red')

    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Number')
    
    plot_title = title
    if n_components_displayed < num_total_features:
        plot_title += f" (First {n_components_displayed} Components)"
    plt.title(plot_title)
    
    plt.xticks(range(1, n_components_displayed + 1))
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print some information
    if n_components_displayed < num_total_features:
        print(f"Showing information for the first {n_components_displayed} principal components (out of {num_total_features} total possible components):")
    else:
        print(f"Showing information for all {n_components_displayed} principal components:")
        
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: Explained Variance = {ratio:.4f}, Cumulative = {cumulative_explained_variance[i]:.4f}")
        # The cumulative_explained_variance[i] is the actual fraction of total variance explained by the first i+1 components.
        if cumulative_explained_variance[i] >= 0.95 and (i == 0 or cumulative_explained_variance[i-1] < 0.95) :
            print(f"--- {i+1} components explain >= 95% of total variance ---")
        if cumulative_explained_variance[i] >= 0.99 and (i == 0 or cumulative_explained_variance[i-1] < 0.99) :
            print(f"--- {i+1} components explain >= 99% of total variance ---")

def plot_overall_results_simple():
    sns.set_palette("muted")

    # Raw results extracted from running each model
    data = {
        'Part': ['Raw Pixels', 'Raw Pixels', 'Sobel Filter', 'Sobel Filter', 'HOG', 'HOG'],
        'PCA': ['Without PCA', 'With PCA', 'Without PCA', 'With PCA', 'Without PCA', 'With PCA'],
        'Accuracy': [0.6442, 0.6362, 0.8349, 0.8462, 0.8301, 0.8189],
        'AUC': [0.5367, 0.5349, 0.9535, 0.9524, 0.9230, 0.9105]
    }
    df = pd.DataFrame(data)

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Part', y='Accuracy', hue='PCA')
    plt.title('Accuracy by Part and PCA')
    plt.ylabel('Accuracy')
    plt.xlabel('Data Input')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot AUC
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Part', y='AUC', hue='PCA')
    plt.title('AUC by Part and PCA')
    plt.ylabel('AUC')
    plt.xlabel('Data Input')
    plt.legend()
    plt.tight_layout()
    plt.show()   

def plot_overall_results_combined():
    sns.set_palette("muted")

    # Raw results extracted from running each model
    data = {
        'Part': ['Raw Pixels', 'Raw Pixels', 'Sobel Filter', 'Sobel Filter', 'HOG', 'HOG'],
        'PCA': ['Without PCA', 'With PCA', 'Without PCA', 'With PCA', 'Without PCA', 'With PCA'],
        'Accuracy': [0.6442, 0.6362, 0.8349, 0.8462, 0.8301, 0.8189],
        'AUC': [0.5367, 0.5349, 0.9535, 0.9524, 0.9230, 0.9105]
    }
    df = pd.DataFrame(data)

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Part', y='Accuracy', hue='PCA')
    plt.title('Accuracy by Part and PCA')
    plt.ylabel('Accuracy')
    plt.xlabel('Data Input')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot AUC
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Part', y='AUC', hue='PCA')
    plt.title('AUC by Part and PCA')
    plt.ylabel('AUC')
    plt.xlabel('Data Input')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison():
    sns.set_palette("muted")
    
    # Data from t-test comparisons
    comparisons = [
        "Part 2 (Sobel) vs Part 1 (Raw)",
        "Part 3 (Histogram) vs Part 1 (Raw)",
        "Part 3 (Histogram) vs Part 2 (Sobel)"
    ]

    accuracy_data = {
        'Comparison': comparisons,
        'Mean Difference': [0.1525, 0.1600, 0.0075],
        'CI_Lower': [0.0971, 0.1155, -0.0384],
        'CI_Upper': [0.2079, 0.2045, 0.0534],
        'p-value': [0.0001536610, 0.0000194007, 0.7202874671],
        'Significant': ['Yes', 'Yes', 'No']
    }

    auc_data = {
        'Comparison': comparisons,
        'Mean Difference': [0.5061, 0.4991, -0.0070],
        'CI_Lower': [0.4171, 0.4096, -0.0188],
        'CI_Upper': [0.5951, 0.5885, 0.0048],
        'p-value': [0.0000004260, 0.0000005004, 0.2116678301],
        'Significant': ['Yes', 'Yes', 'No']
    }

    # Create DataFrames
    df_accuracy = pd.DataFrame(accuracy_data)
    df_auc = pd.DataFrame(auc_data)

    # Calculate error bars (from CI to mean difference)
    df_accuracy['yerr_low'] = df_accuracy['Mean Difference'] - df_accuracy['CI_Lower']
    df_accuracy['yerr_high'] = df_accuracy['CI_Upper'] - df_accuracy['Mean Difference']

    df_auc['yerr_low'] = df_auc['Mean Difference'] - df_auc['CI_Lower']
    df_auc['yerr_high'] = df_auc['CI_Upper'] - df_auc['Mean Difference']

    # Plot Accuracy differences
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_accuracy['Comparison'], df_accuracy['Mean Difference'], 
                    yerr=[df_accuracy['yerr_low'], df_accuracy['yerr_high']], 
                    capsize=5, alpha=0.7)

    # Color bars by significance
    for i, bar in enumerate(bars):
        if df_accuracy['Significant'][i] == 'Yes':
            bar.set_color('green')
        else:
            bar.set_color('gray')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Accuracy Differences Between Models with 95% Confidence Intervals')
    plt.ylabel('Mean Difference in Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Annotate with p-values
    for i, value in enumerate(df_accuracy['Mean Difference']):
        plt.text(i, value + 0.02, f'p={df_accuracy["p-value"][i]:.7f}', 
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Plot AUC differences
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_auc['Comparison'], df_auc['Mean Difference'], 
                    yerr=[df_auc['yerr_low'], df_auc['yerr_high']], 
                    capsize=5, alpha=0.7)

    # Color bars by significance
    for i, bar in enumerate(bars):
        if df_auc['Significant'][i] == 'Yes':
            bar.set_color('green')
        else:
            bar.set_color('gray')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('AUC Differences Between Models with 95% Confidence Intervals')
    plt.ylabel('Mean Difference in AUC')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Annotate with p-values
    for i, value in enumerate(df_auc['Mean Difference']):
        plt.text(i, value + 0.02, f'p={df_auc["p-value"][i]:.7f}', 
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.show()  

def visualize_weight_from_feature_vector(model, image_shape=(224, 224), pixels_per_cell=(32, 32), orientations=9):

    weights = model.coef_[0]

    n_cells_y = image_shape[0] // pixels_per_cell[0]
    n_cells_x = image_shape[1] // pixels_per_cell[1]
    
    expected_len = n_cells_y * n_cells_x * orientations
    if len(weights) != expected_len:
        print(f"Warning: Weight vector length ({len(weights)}) does not match "
              f"expected HOG feature length ({expected_len} = "
              f"{n_cells_y}x{n_cells_x} cells * {orientations} orientations). Cannot visualize weights accurately.")
        return

    try:
        weights_reshaped = weights.reshape((n_cells_y, n_cells_x, orientations))
    except ValueError as e:
        print(f"Error reshaping weights vector of length {len(weights)} "
              f"into ({n_cells_y}, {n_cells_x}, {orientations}). Cannot visualize. Error: {e}")
        return

    cell_importance = np.sum(np.abs(weights_reshaped), axis=2) # Shape: (n_cells_y, n_cells_x)

    upscaled_heatmap = np.kron(cell_importance, np.ones(pixels_per_cell))

    plt.figure(figsize=(10, 10))
    im = plt.imshow(upscaled_heatmap, cmap='viridis', extent=(0, image_shape[1], image_shape[0], 0))
    plt.colorbar(im, label='Aggregated Absolute Weight per Cell')
    plt.title(f'HOG Feature Importance Heatmap ({pixels_per_cell[0]}x{pixels_per_cell[1]} cells)')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Image Height (pixels)')
    
    ax = plt.gca()
    ax.set_xticks(np.arange(0, image_shape[1] + 1, pixels_per_cell[1]), minor=True)
    ax.set_yticks(np.arange(0, image_shape[0] + 1, pixels_per_cell[0]), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=1, alpha=0.5) 
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False, labeltop=False, labelright=False) 

    plt.show()


def visualize_unsigned_orientations(model, image_shape=(224, 224), pixels_per_cell=(32, 32), orientations=9):
    weights = model.coef_[0]

    n_cells_y = image_shape[0] // pixels_per_cell[0]
    n_cells_x = image_shape[1] // pixels_per_cell[1]
    n_cells = n_cells_y * n_cells_x

    expected_len = n_cells * orientations
    if len(weights) != expected_len:
        print(f"Weight vector length mismatch: {len(weights)} != {expected_len}")
        return

    max_weight = np.max(np.abs(weights))
    heatmap = np.zeros((n_cells_y, n_cells_x))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    angle_step = np.pi / orientations  # 0–π for unsigned gradients
    line_half_len = min(pixels_per_cell) * 0.4  # Half-length for symmetric line

    for idx in range(n_cells):
        y = idx // n_cells_x
        x = idx % n_cells_x

        start = idx * orientations
        end = start + orientations
        cell_weights = weights[start:end]

        heatmap[y, x] = np.sum(np.abs(cell_weights))

        center_x = x * pixels_per_cell[1] + pixels_per_cell[1] / 2
        center_y = y * pixels_per_cell[0] + pixels_per_cell[0] / 2

        for ori in range(orientations):
            angle = ori * angle_step
            weight = cell_weights[ori]
            opacity = np.abs(weight) / max_weight if max_weight > 0 else 0

            dx = np.cos(angle) * line_half_len
            dy = -np.sin(angle) * line_half_len  # Flip Y-axis for image display

            # Draw a symmetric line segment centered at the cell
            ax.plot([center_x - dx, center_x + dx],
                    [center_y - dy, center_y + dy],
                    color=(1, 1, 1, opacity), linewidth=1.5)

    upscaled_heatmap = np.kron(heatmap, np.ones(pixels_per_cell))
    im = plt.imshow(upscaled_heatmap, cmap='viridis', extent=(0, image_shape[1], image_shape[0], 0))
    plt.colorbar(im, label='Aggregated Absolute Weight per Cell')

    ax.set_xticks(np.arange(0, image_shape[1] + 1, pixels_per_cell[1]), minor=True)
    ax.set_yticks(np.arange(0, image_shape[0] + 1, pixels_per_cell[0]), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    plt.title('HOG Orientation Visualization (Unsigned, 0–180°)')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Image Height (pixels)')
    plt.show()

def visualize_unsigned_orientations_max(model, image_shape=(224, 224), pixels_per_cell=(32, 32), orientations=9):
    weights = model.coef_[0]

    n_cells_y = image_shape[0] // pixels_per_cell[0]
    n_cells_x = image_shape[1] // pixels_per_cell[1]
    n_cells = n_cells_y * n_cells_x

    expected_len = n_cells * orientations
    if len(weights) != expected_len:
        print(f"Weight vector length mismatch: {len(weights)} != {expected_len}")
        return

    max_weight = np.max(np.abs(weights))
    heatmap = np.zeros((n_cells_y, n_cells_x))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    angle_step = np.pi / orientations  # 0–π for unsigned gradients
    line_half_len = min(pixels_per_cell) * 0.4  # Half-length for symmetric line

    for idx in range(n_cells):
        y = idx // n_cells_x
        x = idx % n_cells_x

        start = idx * orientations
        end = start + orientations
        cell_weights = weights[start:end]

        # Set the heatmap value to the max weight in this cell
        heatmap[y, x] = np.max(np.abs(cell_weights))

        center_x = x * pixels_per_cell[1] + pixels_per_cell[1] / 2
        center_y = y * pixels_per_cell[0] + pixels_per_cell[0] / 2

        for ori in range(orientations):
            angle = ori * angle_step
            weight = cell_weights[ori]
            opacity = np.abs(weight) / max_weight if max_weight > 0 else 0

            dx = np.cos(angle) * line_half_len
            dy = -np.sin(angle) * line_half_len  # Flip Y-axis for image display

            # Draw a symmetric line segment centered at the cell
            ax.plot([center_x - dx, center_x + dx],
                    [center_y - dy, center_y + dy],
                    color=(1, 1, 1, opacity), linewidth=1.5)

    # Upscale the heatmap for better resolution
    upscaled_heatmap = np.kron(heatmap, np.ones(pixels_per_cell))
    
    # Plot the heatmap
    im = plt.imshow(upscaled_heatmap, cmap='viridis', extent=(0, image_shape[1], image_shape[0], 0))
    plt.colorbar(im, label='Max Absolute Weight per Cell')

    ax.set_xticks(np.arange(0, image_shape[1] + 1, pixels_per_cell[1]), minor=True)
    ax.set_yticks(np.arange(0, image_shape[0] + 1, pixels_per_cell[0]), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    plt.title('HOG Orientation Visualization (Unsigned, 0–180°)')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Image Height (pixels)')
    plt.show()
