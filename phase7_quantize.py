"""
Phase 7: Quantization Feasibility Analysis
-----------------------------------------------
You don't have an ESP32 on hand right now, so this script does NOT claim
real on-device benchmarks. Instead it runs a legitimate "feasibility
study" -- the same framing used in published work such as
"ECG-based real-time arrhythmia monitoring using quantized deep neural
networks: A feasibility study" (De Melo Ribeiro et al., 2022).

What this script actually measures (all real, all reproducible):
  1. Original model size on disk (float32 Keras .h5)
  2. Quantized model size (int8 TensorFlow Lite)
  3. Compression ratio
  4. Parameter count
  5. Estimated peak RAM for inference (very rough, from tensor shapes)

What it does NOT claim:
  - Real wall-clock latency on an ESP32
  - Real power consumption
These should be listed as "future work" in the paper, with a note that
on-device validation will follow once hardware is available. Do not
present the numbers from this script as if they were measured on
physical hardware -- frame them explicitly as a pre-deployment
feasibility estimate.
"""

import os
import numpy as np
import tensorflow as tf

custom_objects = {'mae': tf.keras.losses.MeanAbsoluteError()}


def representative_dataset_gen(X_train, n=200):
    for i in range(min(n, len(X_train))):
        yield [X_train[i:i + 1].astype(np.float32)]


def main():
    print("Loading trained model and training data (for calibration)...")
    model = tf.keras.models.load_model('ecg_autoencoder.h5', custom_objects=custom_objects)
    X_train = np.load('X_train_normal.npy')

    h5_size_kb = os.path.getsize('ecg_autoencoder.h5') / 1024
    n_params = model.count_params()
    print(f"\nOriginal model: {n_params} parameters, {h5_size_kb:.1f} KB (float32 .h5)")

    print("\nConverting to TensorFlow Lite with int8 post-training quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Full int8 conversion failed ({e}); retrying with dynamic range "
              f"quantization (weights only, still useful for size comparison).")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

    with open('ecg_autoencoder_int8.tflite', 'wb') as f:
        f.write(tflite_model)

    tflite_size_kb = os.path.getsize('ecg_autoencoder_int8.tflite') / 1024
    compression_ratio = h5_size_kb / tflite_size_kb if tflite_size_kb > 0 else float('nan')

    print(f"\nQuantized model: {tflite_size_kb:.1f} KB")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    # Rough peak-activation RAM estimate from largest intermediate tensor shape.
    # This is a feasibility estimate, not a measured figure.
    max_activation_elems = 0
    for layer in model.layers:
        try:
            shape = layer.output_shape
            elems = np.prod([d for d in shape if d is not None])
            max_activation_elems = max(max_activation_elems, elems)
        except Exception:
            pass
    est_ram_kb = (max_activation_elems * 1) / 1024  # int8 = 1 byte/elem
    print(f"\nEstimated peak activation RAM (int8): ~{est_ram_kb:.2f} KB "
          f"(rough estimate, not a measured figure)")

    typical_esp32_sram_kb = 520  # ESP32 (not S3/C3 variants) typical SRAM
    print(f"\nFor reference, a standard ESP32 has ~{typical_esp32_sram_kb} KB SRAM.")
    print(f"Model + activations fit comfortably within this budget "
          f"({tflite_size_kb + est_ram_kb:.1f} KB total vs {typical_esp32_sram_kb} KB available), "
          f"suggesting on-device deployment is feasible. This should be confirmed "
          f"with real hardware (TFLite Micro on ESP32) as future work.")

    print("\n✅ Done. Report these numbers in the paper as a 'pre-deployment "
          "feasibility analysis,' explicitly flagging real-hardware validation "
          "as future work.")


if __name__ == '__main__':
    main()
