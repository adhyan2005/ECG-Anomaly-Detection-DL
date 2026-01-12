import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

# 1. Load the preprocessed data (Normal beats only)
print("Loading preprocessed heartbeats...")
X_train = np.load('X_normal.npy')

# 2. Build the LSTM Autoencoder
model = Sequential([
    # Encoder
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    RepeatVector(X_train.shape[1]),
    # Decoder
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2]))
])

model.compile(optimizer='adam', loss='mae')
model.summary()

# 3. Train the model
print("\nStarting Training (20 Epochs)... This may take 3-5 mins on your Mac.")
history = model.fit(
    X_train, X_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# 4. Save the trained model
model.save('ecg_autoencoder.h5')
print("\n✅ SUCCESS: Model saved as 'ecg_autoencoder.h5'")

# 5. Save the training progress chart
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("AI Learning Progress (Loss Curve)")
plt.xlabel("Epochs")
plt.ylabel("Error (MAE)")
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
print("✅ Progress chart saved as 'training_progress.png'")
