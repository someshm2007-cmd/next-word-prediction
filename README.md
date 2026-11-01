# 🤖 Next Word Predictor using Deep Learning

A Natural Language Processing (NLP) project built with **TensorFlow and Keras** that predicts subsequent words based on a given sequence[cite: 3]. The model is powered by **GRU (Gated Recurrent Unit)** layers and trained on news headline data[cite: 3].

---

## 📌 Project Highlights
* **Dataset:** Million News Headlines (`abcnews-date-text.csv`)[cite: 3]
* **Preprocessing & Tokenization:** Text sequences are tokenized, limited to a vocabulary of top 8,000 words, and padded for uniform input length[cite: 3, 8].
* **Model Architecture:** Sequential network featuring Embedding layers, Spatial Dropout, stacked GRU layers, and a Softmax Dense output layer[cite: 3, 6].
* **Callbacks:** Includes `EarlyStopping` and `ReduceLROnPlateau` to monitor validation loss and optimize model training[cite: 3, 6].
* **Text Generation:** Implements custom temperature-based sampling (`sample_with_temperature`) to generate realistic word sequences[cite: 3].

---

## 🛠️ Tech Stack
* **Python**[cite: 3]
* **TensorFlow / Keras**[cite: 3]
* **Pandas & NumPy**[cite: 3]
* **Matplotlib** (for accuracy/loss visualization)[cite: 3]

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
```[cite: 3]

---

## 📈 Training Configuration
* **Loss Function:** `sparse_categorical_crossentropy`[cite: 3]
* **Optimizer:** `Adam` (Learning Rate = `0.001`)[cite: 3]
* **Batch Size:** `128`[cite: 3] by providing a seed text:
```python
text = "waht are"
# The model iteratively predicts and appends subsequent words using temperature-controlled sampling.
```[cite: 3]
