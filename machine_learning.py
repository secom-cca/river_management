import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# ------------------------------
# 1. AMeDAS（久留米）の整形
# ------------------------------
merged_df = pd.read_csv("data/merged_df.csv", parse_dates=["date"])

# ------------------------------
# 5. 時系列データセットの作成
# ------------------------------
def make_timeseries_dataset(df, input_features, output_features, sequence_length=7):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        input_seq = df[input_features].iloc[i:i + sequence_length].values
        output_seq = df[output_features].iloc[i + sequence_length].values
        X.append(input_seq)
        y.append(output_seq)
    return np.array(X), np.array(y)

input_features = ['precipitation', 'temperature']
output_features = ['flow', 'level']
X, y = make_timeseries_dataset(merged_df, input_features, output_features)

# ------------------------------
# 6. スケーリング
# ------------------------------
scaler_X = StandardScaler()
# X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
X_scaled = np.zeros_like(X)
for i in range(X.shape[2]):
    X_scaled[:, :, i] = StandardScaler().fit_transform(X[:, :, i])


scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# ------------------------------
# 7. 学習データと検証データに分割
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# ------------------------------
# 8. LSTM モデル構築・学習
# ------------------------------
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(output_features))  # 出力は flow, level
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ------------------------------
# 9. 予測と可視化
# ------------------------------
y_pred_scaled = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_val)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(y_true[:, 0], label='True Flow')
plt.plot(y_pred[:, 0], label='Predicted Flow')
plt.title('Flow Prediction')
plt.xlabel('Time Step')
plt.ylabel('Flow')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true[:, 1], label='True Level')
plt.plot(y_pred[:, 1], label='Predicted Level')
plt.title('Level Prediction')
plt.xlabel('Time Step')
plt.ylabel('Level')
plt.legend()

plt.tight_layout()
plt.show()
