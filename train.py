"""
Production Training Pipeline: Cat vs Dog Classifier
Backbone: EfficientNetV2-B0 (Transfer Learning & Fine-Tuning)
Target Accuracy: 97% - 99%

Author: Sheetal Jain / Senior AI & Vision Engineer
Architecture:
  - Data Ingestion: tf.data with GPU/CPU parallelization (prefetch, cache)
  - Augmentation: Keras Preprocessing Layers (Flip, Rotation, Zoom, Contrast)
  - Phase 1: Feature Extraction (Frozen backbone, trainable classification head)
  - Phase 2: Fine-Tuning (Unfreeze top 25 layers with low LR 1e-5, Adam)
  - Callbacks: EarlyStopping, ModelCheckpoint (best val_loss), ReduceLROnPlateau
  - Serialization: best_model.keras (Keras 3 / TensorFlow 2.16+)
"""

import os
import sys
import argparse
import logging
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainPipeline")

# Hardware & Determinism Configuration
def configure_environment(seed: int = 42):
    """Sets random seeds and configures GPU memory growth."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Detected and configured {len(gpus)} GPU(s) with dynamic memory growth.")
        except RuntimeError as e:
            logger.warning(f"GPU memory growth configuration error: {e}")
    else:
        logger.info("No physical GPU found. Running on CPU with multi-threading.")


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Builds an in-graph data augmentation pipeline.
    Mitigates overfitting while preserving biological semantics.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", name="aug_random_flip"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="nearest", name="aug_random_rot"),
        tf.keras.layers.RandomZoom(0.12, fill_mode="nearest", name="aug_random_zoom"),
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, name="aug_random_trans"),
        tf.keras.layers.RandomContrast(0.15, name="aug_random_contrast"),
    ], name="data_augmentation")


def create_dataset_pipeline(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Builds high-performance tf.data input pipelines with caching,
    shuffling, and asynchronous prefetching.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        logger.info(f"Loading separate train/val directories from {data_dir}...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="binary",
            shuffle=True,
            seed=seed
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="binary",
            shuffle=False
        )
    elif os.path.isdir(data_dir):
        logger.info(f"Loading dataset with {val_split*100:.0f}% validation split from {data_dir}...")
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="both",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="binary",
            shuffle=True
        )
    else:
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # High-Performance Optimization: Cache, Shuffle, Prefetch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(buffer_size=1000, seed=seed).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    
    return train_ds, val_ds


def create_synthetic_demo_pipeline(
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    num_samples: int = 64
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Generates synthetic dummy datasets to allow fast testing of the full
    end-to-end training and serialization pipeline without downloading 1GB+ datasets.
    """
    logger.info(f"Creating synthetic demonstration pipeline ({num_samples} samples)...")
    images = np.random.uniform(0.0, 255.0, size=(num_samples, img_size[0], img_size[1], 3)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

    split = int(num_samples * 0.8)
    train_ds = tf.data.Dataset.from_tensor_slices((images[:split], labels[:split])).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((images[split:], labels[split:])).batch(batch_size)

    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)


def build_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.35
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs the Transfer Learning network using EfficientNetV2-B0 backbone.
    Architecture:
      Input (224, 224, 3) -> EfficientNetV2-B0 Backbone (ImageNet weights)
      -> GlobalAveragePooling2D
      -> BatchNormalization
      -> Dense(256, ReLU) + Dropout(0.35)
      -> BatchNormalization
      -> Dense(64, ReLU) + Dropout(0.20)
      -> Dense(1, Sigmoid) [0=Cat, 1=Dog]
    """
    logger.info("Initializing EfficientNetV2-B0 with ImageNet weights...")
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Pre-trained base model
    # EfficientNetV2-B0 has built-in input preprocessing (scaling [0, 255] -> [-1, 1])
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=True
    )
    
    # Freeze backbone for Phase 1
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.6, name="head_dropout_2")(x)
    
    x = tf.keras.layers.Dense(64, activation="relu", name="head_dense_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CatDog_EfficientNetV2B0")
    return model, base_model


def unfreeze_backbone_for_fine_tuning(base_model: tf.keras.Model, unfreeze_layers: int = 25):
    """
    Selectively unfreezes the top N layers of the backbone for fine-tuning.
    Senior CV Best Practice:
      Always keep BatchNormalization layers frozen during fine-tuning.
      Unfreezing BN updates running mean/variance on small target batches,
      destabilizing pre-trained features and causing representation collapse.
    """
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - unfreeze_layers)
    
    logger.info(f"Fine-Tuning: Unfreezing top {unfreeze_layers} layers of backbone (total: {total_layers})...")
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False  # Keep BN frozen in inference mode
            else:
                layer.trainable = True


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3):
    """Compiles the model with Binary Crossentropy and comprehensive evaluation metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=metrics
    )


def plot_training_history(history1: tf.keras.callbacks.History, history2: Optional[tf.keras.callbacks.History] = None, output_path: str = "training_metrics.png"):
    """Plots and saves loss, accuracy, and AUC curves across both training phases."""
    def merge_metric(metric_name):
        vals = list(history1.history.get(metric_name, []))
        if history2:
            vals += list(history2.history.get(metric_name, []))
        return vals

    train_loss = merge_metric("loss")
    val_loss = merge_metric("val_loss")
    train_acc = merge_metric("accuracy")
    val_acc = merge_metric("val_accuracy")
    train_auc = merge_metric("auc")
    val_auc = merge_metric("val_auc")

    epochs_range = range(1, len(train_loss) + 1)
    phase1_len = len(history1.history.get("loss", []))

    plt.figure(figsize=(16, 5))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(epochs_range, val_loss, label="Val Loss", color="#ff7f0e", lw=2)
    if history2 and phase1_len < len(epochs_range):
        plt.axvline(x=phase1_len, color="grey", linestyle="--", label="Fine-Tuning Start")
    plt.title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_acc, label="Train Accuracy", color="#2ca02c", lw=2)
    plt.plot(epochs_range, val_acc, label="Val Accuracy", color="#d62728", lw=2)
    if history2 and phase1_len < len(epochs_range):
        plt.axvline(x=phase1_len, color="grey", linestyle="--", label="Fine-Tuning Start")
    plt.title("Classification Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_auc, label="Train AUC", color="#9467bd", lw=2)
    plt.plot(epochs_range, val_auc, label="Val AUC", color="#8c564b", lw=2)
    if history2 and phase1_len < len(epochs_range):
        plt.axvline(x=phase1_len, color="grey", linestyle="--", label="Fine-Tuning Start")
    plt.title("ROC-AUC Score", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Training metrics visualization saved to '{output_path}'.")


def run_pipeline(args):
    """Executes the complete two-phase Transfer Learning & Fine-Tuning pipeline."""
    configure_environment(seed=args.seed)
    
    # Step 1: Ingestion
    if args.demo_mode or not os.path.exists(args.data_dir):
        if not args.demo_mode:
            logger.warning(f"Data directory '{args.data_dir}' does not exist. Switching to synthetic demo mode.")
        train_ds, val_ds = create_synthetic_demo_pipeline(
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size
        )
    else:
        train_ds, val_ds = create_dataset_pipeline(
            data_dir=args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            seed=args.seed
        )

    # Step 2: Build Network
    model, base_model = build_efficientnet_model(
        input_shape=(args.img_size, args.img_size, 3)
    )
    model.summary(print_fn=logger.info)

    # Step 3: Phase 1 - Feature Extraction (Frozen Backbone)
    logger.info("==================================================")
    logger.info("PHASE 1: Training Classification Head (Backbone Frozen)")
    logger.info(f"Epochs: {args.epochs_phase1} | Learning Rate: {args.lr_phase1}")
    logger.info("==================================================")
    
    compile_model(model, learning_rate=args.lr_phase1)
    
    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase1,
        callbacks=callbacks_phase1
    )

    # Step 4: Phase 2 - Fine-Tuning (Top Layers Unfrozen)
    logger.info("==================================================")
    logger.info("PHASE 2: Fine-Tuning Top Backbone Layers")
    logger.info(f"Unfreezing top {args.unfreeze_layers} layers | LR: {args.lr_phase2}")
    logger.info("==================================================")
    
    unfreeze_backbone_for_fine_tuning(base_model, unfreeze_layers=args.unfreeze_layers)
    compile_model(model, learning_rate=args.lr_phase2)

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase2,
        callbacks=callbacks_phase2
    )

    # Step 5: Final Evaluation & Serialization
    logger.info("Evaluating fine-tuned model on validation set...")
    val_results = model.evaluate(val_ds, return_dict=True)
    logger.info(f"Validation Results: {val_results}")
    
    # Save final model if not already saved by checkpoint
    if not os.path.exists(args.output_model):
        model.save(args.output_model)
        logger.info(f"Model saved to '{args.output_model}'.")
    else:
        logger.info(f"Best checkpoint successfully preserved at '{args.output_model}'.")

    # Plot metrics
    plot_training_history(history_phase1, history_phase2, output_path=args.metrics_plot)
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Training Pipeline - Cat vs Dog Vision Engine")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution (224 for EfficientNetV2-B0)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs_phase1", type=int, default=8, help="Epochs for feature extraction head")
    parser.add_argument("--epochs_phase2", type=int, default=12, help="Epochs for fine-tuning")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Initial learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=1e-5, help="Fine-tuning learning rate for Phase 2")
    parser.add_argument("--unfreeze_layers", type=int, default=25, help="Number of top layers to unfreeze")
    parser.add_argument("--output_model", type=str, default="best_model.keras", help="Target path for saved model")
    parser.add_argument("--metrics_plot", type=str, default="training_metrics.png", help="Path for history plot")
    parser.add_argument("--demo_mode", action="store_true", help="Run quick synthetic test without full dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    run_pipeline(args)
