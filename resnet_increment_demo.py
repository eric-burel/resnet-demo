"""Train a minimal residual network to learn the mapping x -> x + 1.

This script mirrors ``increment_demo.py`` but uses a tiny ResNet-style
architecture (skip connection) on normalized scalar inputs.
"""

import numpy as np
import tensorflow as tf


def build_dataset(num_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized inputs and targets for the mapping x -> x + 1."""
    raw_inputs = rng.integers(low=0, high=255, size=(num_samples,), dtype=np.int64)
    inputs = raw_inputs.astype(np.float32) / 255.0
    targets = (raw_inputs + 1).astype(np.float32) / 255.0
    return inputs[:, None], targets[:, None]


def build_resnet_model() -> tf.keras.Model:
    """Create a 1D residual model where the skip path carries the input forward."""
    inputs = tf.keras.Input(shape=(1,), name="scalar_input")

    # Small residual branch: project through a hidden layer and back to 1D.
    x = tf.keras.layers.Dense(8, activation="relu", name="residual_dense_1")(inputs)
    residual = tf.keras.layers.Dense(1, activation=None, name="residual_dense_2")(x)

    outputs = tf.keras.layers.Add(name="skip_add")([inputs, residual])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_increment")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
        loss="mse",
        metrics=["mae"],
    )
    return model


def denormalize(values: np.ndarray) -> np.ndarray:
    """Convert normalized predictions back to the [0, 255] scale."""
    return values * 255.0


def print_final_weights(model: tf.keras.Model) -> None:
    """Display learned weights for each trainable layer."""
    print("\nFinal weights:")
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        print(f"  {layer.name}:")
        for idx, weight in enumerate(weights):
            squeezed = np.squeeze(weight)
            print(f"    weight[{idx}] shape={weight.shape} values={squeezed}")


def main() -> None:
    rng = np.random.default_rng(seed=123)
    train_inputs, train_targets = build_dataset(num_samples=64, rng=rng)
    test_inputs, test_targets = build_dataset(num_samples=32, rng=rng)

    model = build_resnet_model()
    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(test_inputs, test_targets),
        epochs=30,
        batch_size=16,
        verbose=0,
    )

    test_loss, test_mae = model.evaluate(test_inputs, test_targets, verbose=0)
    print("Test results:")
    print(f"  MSE: {test_loss:.6f}")
    print(f"  MAE: {test_mae:.6f}")

    sample_inputs = test_inputs[:5]
    sample_targets = test_targets[:5]
    predictions = model.predict(sample_inputs, verbose=0)
    denorm_inputs = denormalize(sample_inputs.squeeze())
    denorm_targets = denormalize(sample_targets.squeeze())
    denorm_predictions = denormalize(predictions.squeeze())

    print("\nSample predictions (values rounded to 3 decimals):")
    for inp, target, pred in zip(denorm_inputs, denorm_targets, denorm_predictions):
        print(f"  input: {inp:6.3f} -> target: {target:6.3f}, predicted: {pred:6.3f}")

    print("\nTraining history (last 5 epochs):")
    for epoch, (loss, val_loss) in enumerate(
        zip(history.history["loss"][-5:], history.history["val_loss"][-5:]),
        start=len(history.history["loss"]) - 5,
    ):
        print(f"  Epoch {epoch + 1:2d}: loss={loss:.6f}, val_loss={val_loss:.6f}")

    print_final_weights(model)


if __name__ == "__main__":
    main()
