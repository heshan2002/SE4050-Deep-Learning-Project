import gc
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def ensure_dataset():
    archive_path = DATASET_DIR / "realwaste.zip"
    dataset_path = DATASET_DIR / "realwaste" / "realwaste-main" / "RealWaste"

    if dataset_path.exists():
        return dataset_path

    if not archive_path.exists():
        print("Downloading RealWaste dataset...")
        url = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
        urllib.request.urlretrieve(url, archive_path)

    print("Extracting RealWaste dataset...")
    extraction_root = DATASET_DIR / "realwaste"
    extraction_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extraction_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found at expected path: {dataset_path}")

    return dataset_path


def prepare_dataframe(df, dataset_path):
    df = df.copy()
    df["filepath"] = df["relative_path"].apply(lambda x: str(dataset_path / x))
    class_names = sorted(df["label"].unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    df["label_index"] = df["label"].map(class_to_index)
    return df, class_names


def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label


def create_dataset(df, training=False):
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label_index"].values))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])


def evaluate_model(model, model_name, test_ds, class_names):
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predictions = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{model_name} Test Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / f"{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(11, 10))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results_df = pd.DataFrame([
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Test Loss": test_loss,
        }
    ])

    results_df.to_csv(RESULTS_DIR / f"{model_name}_results.csv", index=False)
    return results_df


def main():
    dataset_path = ensure_dataset()

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    train_df, class_names = prepare_dataframe(train_df, dataset_path)
    val_df, _ = prepare_dataframe(val_df, dataset_path)
    test_df, _ = prepare_dataframe(test_df, dataset_path)

    NUM_CLASSES = len(class_names)
    print("Class names:", class_names)
    print("Number of classes:", NUM_CLASSES)
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    train_ds = create_dataset(train_df, training=True)
    val_ds = create_dataset(val_df)
    test_ds = create_dataset(test_df)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]

    tf.keras.backend.clear_session()
    gc.collect()

    efficient_base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    efficient_base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = get_augmentation()(inputs)
    x = efficient_base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    efficient_model = tf.keras.Model(inputs, outputs)
    efficient_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    efficient_history = efficient_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(efficient_history.history["accuracy"], label="Training Accuracy")
    plt.plot(efficient_history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNetB0 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "EfficientNetB0_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(efficient_history.history["loss"], label="Training Loss")
    plt.plot(efficient_history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNetB0 Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "EfficientNetB0_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    efficient_base.trainable = True
    for layer in efficient_base.layers[:-30]:
        layer.trainable = False
    for layer in efficient_base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    efficient_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    efficient_fine_history = efficient_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks,
    )

    efficient_results = evaluate_model(efficient_model, "EfficientNetB0", test_ds, class_names)
    efficient_model.save(RESULTS_DIR / "EfficientNetB0.keras")
    print("\nSaved results to:", RESULTS_DIR)
    print(efficient_results.to_string(index=False))


if __name__ == "__main__":
    main()
