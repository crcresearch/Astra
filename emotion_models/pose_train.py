import os
import sys
import numpy as np
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import argparse

# Import our PoseFeatureExtractor class
from extract import PoseFeatureExtractor

# Constants
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def find_json_directories(data_dir):
    """
    Find directories containing JSON files with pose data
    
    Args:
        data_dir: Root data directory
        
    Returns:
        List of directories containing JSON files
    """
    json_dirs = []
    
    # Walk through the data directory
    for root, dirs, files in os.walk(data_dir):
        # Check if this directory contains JSON files
        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            json_dirs.append(root)
    
    return json_dirs

def determine_emotion_from_path(path):
    """
    Determine emotion from directory or file path
    
    Args:
        path: Directory or file path
        
    Returns:
        Emotion name or None
    """
    path_lower = path.lower()
    
    # Check for emotion names in the path
    for emotion in EMOTIONS:
        if emotion.lower() in path_lower:
            return emotion
    
    # Check for emotion codes in the path
    emotion_codes = {
        'A': 'Anger',
        'D': 'Disgust',
        'F': 'Fear',
        'H': 'Happiness',
        'N': 'Neutral',
        'SA': 'Sadness',
        'SU': 'Surprise'
    }
    
    for code, emotion in emotion_codes.items():
        # Check for code with word boundaries or surrounded by non-alphanumeric
        if f"_{code}" in path or f"{code}_" in path or f"{code}0" in path or f"{code}1" in path or f"{code}2" in path:
            return emotion
    
    return None

def extract_features_from_directories(json_dirs, feature_extractor):
    """
    Extract features from directories containing JSON files
    
    Args:
        json_dirs: List of directories containing JSON files
        feature_extractor: PoseFeatureExtractor instance
        
    Returns:
        Features and labels
    """
    features = []
    labels = []
    valid_dirs = []
    
    print(f"Extracting features from {len(json_dirs)} directories...")
    
    # First, determine emotions from paths
    for json_dir in json_dirs:
        emotion = determine_emotion_from_path(json_dir)
        if emotion:
            valid_dirs.append((json_dir, emotion))
    
    print(f"Found {len(valid_dirs)} valid directories with identifiable emotions")
    
    # Extract features from each directory with a progress bar
    for json_dir, emotion in tqdm(valid_dirs, desc="Extracting features"):
        try:
            # Extract features from JSON files
            dir_features = feature_extractor.extract_features_from_json_files(json_dir)
            
            if dir_features is not None and len(dir_features) > 0:
                features.append(dir_features)
                labels.append(emotion)
        except Exception as e:
            print(f"Error extracting features from {json_dir}: {e}")
    
    if not features:
        print("No valid features extracted from any directory")
        return None, None
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Extracted {features.shape[1]} features from {len(labels)} recordings")
    print(f"Emotion distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return features, labels

def train_emotion_model(features, labels, model_path, output_dir=None):
    """
    Train an emotion classification model with advanced features
    
    Args:
        features: Extracted features
        labels: Emotion labels
        model_path: Path to save the model
        output_dir: Directory to save visualizations
        
    Returns:
        Trained model, scaler, and label encoder
    """
    if features is None or labels is None:
        print("Error: Features or labels are None")
        return None, None, None
    
    if len(features) == 0:
        print("Error: No features to train on")
        return None, None, None
    
    try:
        print("Training emotion model...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Print label mapping
        label_mapping = {emotion: i for i, emotion in enumerate(label_encoder.classes_)}
        print(f"Label mapping: {label_mapping}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        
        # Try cross-validation first to estimate model performance
        print("Performing cross-validation...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train the final model on all training data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create confusion matrix visualization
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save the plot
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            print(f"Confusion matrix saved to {cm_path}")
            
            # Create feature importance visualization
            feature_importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(feature_importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), feature_importances[indices])
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            
            # Save the plot
            fi_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(fi_path)
            plt.close()
            
            print(f"Feature importance plot saved to {fi_path}")
        
        # Create the model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model, scaler, and encoder
        with open(model_path, 'wb') as f:
            pickle.dump((model, scaler, label_encoder), f)
        
        print(f"Model saved to {model_path}")
        
        return model, scaler, label_encoder
    
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_csv_data(csv_path):
    """
    Load data from CSV file if available
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Features and labels
    """
    try:
        print(f"Loading data from CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have emotion labels
        if 'emotion' in df.columns:
            labels = df['emotion'].values
            print(f"Found emotion labels: {np.unique(labels)}")
            
            # Check if we have advanced features
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            
            if feature_cols:
                print(f"Found {len(feature_cols)} feature columns")
                features = df[feature_cols].values
                
                return features, labels
        
        print("CSV does not contain expected advanced features and labels")
        return None, None
    
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None

def save_features_to_csv(features, labels, output_path):
    """
    Save extracted features to CSV file
    
    Args:
        features: Extracted features
        labels: Emotion labels
        output_path: Path to save CSV file
    """
    try:
        # Create DataFrame with features
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_cols)
        
        # Add label column
        df['emotion'] = labels
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"Features saved to {output_path}")
    
    except Exception as e:
        print(f"Error saving features to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model with advanced pose features')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing pose data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model and visualizations')
    parser.add_argument('--force_extract', action='store_true', help='Force feature extraction even if CSV exists')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    model_path = os.path.join(args.output_dir, 'advanced_emotion_model.pkl')
    features_csv_path = os.path.join(args.output_dir, 'advanced_features.csv')
    
    # Check if we already have extracted features
    features = None
    labels = None
    
    if os.path.exists(features_csv_path) and not args.force_extract:
        print(f"Found existing features file: {features_csv_path}")
        features, labels = load_csv_data(features_csv_path)
    
    # If no existing features or force_extract, extract new features
    if features is None or labels is None or args.force_extract:
        print("Extracting new features...")
        
        # Initialize feature extractor
        feature_extractor = PoseFeatureExtractor()
        
        # Find directories with JSON files
        json_dirs = find_json_directories(args.data_dir)
        print(f"Found {len(json_dirs)} directories with JSON files")
        
        if not json_dirs:
            print(f"Error: No directories with JSON files found in {args.data_dir}")
            return
        
        # Extract features
        features, labels = extract_features_from_directories(json_dirs, feature_extractor)
        
        if features is not None and labels is not None:
            # Save features to CSV for future use
            save_features_to_csv(features, labels, features_csv_path)
    
    # Train model
    if features is not None and labels is not None:
        train_emotion_model(features, labels, model_path, args.output_dir)
    else:
        print("Error: Could not extract or load features and labels")

if __name__ == "__main__":
    main()
