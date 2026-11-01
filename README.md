# 🤖 Next Word Predictor using Deep Learning

A Natural Language Processing (NLP) project built with **TensorFlow and Keras** that predicts subsequent words based on a given sequence. The model is powered by **GRU (Gated Recurrent Unit)** layers and trained on news headline data.

---

## 📌 Project Highlights
* **Dataset:** Million News Headlines (`abcnews-date-text.csv`)
* **Preprocessing & Tokenization:** Text sequences are tokenized, limited to a vocabulary of top 8,000 words, and padded for uniform input length.
* **Model Architecture:** Sequential network featuring Embedding layers, Spatial Dropout, stacked GRU layers, and a Softmax Dense output layer.
* **Callbacks:** Includes `EarlyStopping` and `ReduceLROnPlateau` to monitor validation loss and optimize model training.
* **Text Generation:** Implements custom temperature-based sampling (`sample_with_temperature`) to generate realistic word sequences.

---

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras**
* **Pandas & NumPy**
* **Matplotlib** (for accuracy/loss visualization)

---

## 🚀 Model Structure
```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_len,)),
    tf.keras.layers.Embedding(input_dim=total_words, output_dim=embed_dim),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(256, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(total_words, activation="softmax")
])

📈 Training Configuration
Loss Function: sparse_categorical_crossentropy

Optimizer: Adam (Learning Rate = 0.001)

Batch Size: 128

text = "waht are"
# The model iteratively predicts and appends subsequent words using temperature-controlled sampling.
