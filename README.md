# resnet-demo

## Increment demo

This repository includes minimal TensorFlow examples that learn to output `x + 1` for inputs in `[0, 255]`.

### Running the demo

Use the [uv](https://github.com/astral-sh/uv) project manager to install dependencies and run the script:

```bash
uv run python increment_demo.py
# or run the ResNet-style variant
uv run python resnet_increment_demo.py
```

Each script prints test metrics, sample predictions, and the final learned weights for every trainable layer so you can inspect the parameters directly.
