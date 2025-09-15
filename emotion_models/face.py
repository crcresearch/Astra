import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

# Optional: force CPU mode
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(num_classes, img_size=224):
    """
    Create a MobileNetV2-based model for emotion classification
    
    Args:
        num_classes: Number of emotion classes
        img_size: Input image size
        
    Returns:
        Compiled model
    """
    # Create base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Store base model as custom attribute for fine-tuning later
    model._base_model = base_model
    
    return model

def unfreeze_model(model, num_layers=30):
    """
    Unfreeze the last layers of the base model for fine-tuning
    
    Args:
        model: Model to fine-tune
        num_layers: Number of layers to unfreeze
        
    Returns:
        Fine-tuned model
    """
    # Check if we stored the base model as an attribute
    if hasattr(model, '_base_model'):
        base_model = model._base_model
    else:
        # If no attribute, try to find the base model in the layers
        # This is a fallback approach
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                base_model = layer
                break
        else:
            # If we still can't find it, just make all layers trainable
            print("Base model not found, making all layers trainable")
            model.trainable = True
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
    
    # Unfreeze the base model for fine-tuning
    base_model.trainable = True
    
    # Optionally freeze the initial layers
    if num_layers < len(base_model.layers):
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model_for_finetuning(model_path, num_classes):
    """
    Load a saved model and prepare it for fine-tuning
    
    Args:
        model_path: Path to the saved model
        num_classes: Number of classes
        
    Returns:
        Loaded model ready for fine-tuning
    """
    model = load_model(model_path)
    
    # Try to find the base model within the loaded model
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            model._base_model = layer
            break
    
    return model

def preprocess_image(img_path, img_size=224):
    """
    Load and preprocess a single image
    
    Args:
        img_path: Path to the image
        img_size: Target size
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image
        img = load_img(img_path, target_size=(img_size, img_size))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def load_all_data(image_paths, labels, base_path="", img_size=224):
    """
    Load all images into memory at once (for smaller datasets)
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        base_path: Base path to append to image paths
        img_size: Target image size
        
    Returns:
        Numpy arrays of images and labels
    """
    # Initialize arrays
    valid_images = []
    valid_labels = []
    
    # Load all images
    print("Loading images...")
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} images")
            
        full_path = os.path.join(base_path, img_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"Warning: Image file not found: {full_path}")
            continue
            
        # Preprocess image
        img_tensor = preprocess_image(full_path, img_size)
        
        if img_tensor is not None:
            valid_images.append(img_tensor)
            valid_labels.append(labels[i])
    
    # Convert to numpy arrays
    if valid_images:
        images_array = np.array(valid_images)
        labels_array = np.array(valid_labels)
        return images_array, labels_array
    else:
        return None, None

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training history saved to training_history.png")

def evaluate_model(model, X_val, y_val, class_names):
    """
    Evaluate model and create visualizations
    
    Args:
        model: Trained model
        X_val: Validation images
        y_val: Validation labels
        class_names: List of class names
    """
    # Get predictions
    preds = model.predict(X_val)
    pred_classes = np.argmax(preds, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, pred_classes, target_names=class_names))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train facial emotion recognition model with fixed fine-tuning')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--base_path', type=str, default='', help='Base path for images')
    parser.add_argument('--img_size', type=int, default=160, help='Image size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=5, help='Number of fine-tuning epochs')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model')
    parser.add_argument('--save_path', type=str, default='emotion_model.h5', help='Path to save model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to existing checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Check if data CSV exists
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Data CSV file not found: {args.data_csv}")
    
    print(f"Loading data from {args.data_csv}...")
    
    # Load data CSV
    try:
        df = pd.read_csv(args.data_csv)
        print(f"CSV loaded with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head())
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
    
    # Check for required columns
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'path' and 'label' columns")
    
    # Get unique classes
    class_names = sorted(df['label'].unique())
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['label'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df['path'].values, labels_encoded, 
        test_size=0.2, random_state=42, 
        stratify=labels_encoded
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Load all images into memory
    print("Loading training images...")
    X_train_data, y_train_data = load_all_data(
        X_train, y_train,
        base_path=args.base_path,
        img_size=args.img_size
    )
    
    print("Loading validation images...")
    X_val_data, y_val_data = load_all_data(
        X_val, y_val,
        base_path=args.base_path,
        img_size=args.img_size
    )
    
    if X_train_data is None or X_val_data is None:
        raise ValueError("Failed to load images")
    
    print(f"Loaded {X_train_data.shape[0]} training images and {X_val_data.shape[0]} validation images")
    
    # Create or load model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        model = load_model_for_finetuning(args.checkpoint_path, num_classes)
    else:
        print("Creating new model...")
        model = create_model(num_classes, img_size=args.img_size)
    
    model.summary()
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train initial model if not using checkpoint
    if not args.checkpoint_path:
        print(f"Training initial model for {args.epochs} epochs...")
        history1 = model.fit(
            X_train_data, y_train_data,
            epochs=args.epochs,
            validation_data=(X_val_data, y_val_data),
            callbacks=callbacks,
            verbose=1
        )
    
    # Fine-tune if requested
    if args.fine_tune:
        print("Fine-tuning model...")
        model = unfreeze_model(model)
        
        # Number of epochs to fine-tune
        if args.checkpoint_path:
            # If loading from checkpoint, only do fine-tuning epochs
            fine_tune_start_epoch = 0
            total_epochs = args.fine_tune_epochs
        else:
            # If training from scratch, continue from initial training
            fine_tune_start_epoch = args.epochs
            total_epochs = args.epochs + args.fine_tune_epochs
        
        print(f"Fine-tuning from epoch {fine_tune_start_epoch} to {total_epochs}...")
        
        history2 = model.fit(
            X_train_data, y_train_data,
            epochs=total_epochs,
            initial_epoch=fine_tune_start_epoch,
            validation_data=(X_val_data, y_val_data),
            callbacks=callbacks,
            verbose=1
        )
        
        # If we started from a checkpoint, we only have history2
        if args.checkpoint_path:
            history = history2.history
        else:
            # Combine histories
            history = {}
            history['accuracy'] = history1.history['accuracy'] + history2.history['accuracy']
            history['val_accuracy'] = history1.history['val_accuracy'] + history2.history['val_accuracy']
            history['loss'] = history1.history['loss'] + history2.history['loss']
            history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
    else:
        # If not fine-tuning and using checkpoint
        if args.checkpoint_path:
            print(f"Training loaded model for {args.epochs} epochs...")
            history = model.fit(
                X_train_data, y_train_data,
                epochs=args.epochs,
                validation_data=(X_val_data, y_val_data),
                callbacks=callbacks,
                verbose=1
            ).history
        else:
            history = history1.history
    
    # Plot training history
    plot_training_history(type('obj', (object,), {"history": history}))
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_val_data, y_val_data, class_names)
    
    # Save model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    
    # Save class names
    class_names_file = os.path.splitext(args.save_path)[0] + "_classes.txt"
    with open(class_names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to {class_names_file}")

# Option to resume fine-tuning from a specific checkpoint
if __name__ == "__main__":
    main()
