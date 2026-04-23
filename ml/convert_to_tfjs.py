"""
Convert .keras model to TensorFlow.js format for browser inference.
"""

import tensorflow as tf
from pathlib import Path

model_path = r'c:\CODING\yanoey\artifacts\mammogram_classifier.keras'
output_dir = r'c:\CODING\yanoey\web\model'

# Load model
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Convert to TFJS
print(f"Converting to TensorFlow.js format...")
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, output_dir)

print(f"Model saved to {output_dir}")
print("\nGenerated files:")
for f in Path(output_dir).iterdir():
    print(f"  - {f.name}")
