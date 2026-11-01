
# 🤖 Next Word Predictor using Deep Learning

A natural language processing (NLP) model built with **TensorFlow and Keras** to predict the next word in a sequence. The project utilizes **Gated Recurrent Unit (GRU)** layers and is trained on a subset of the **Million News Headlines** dataset.

---

## 📌 Project Overview
* **Dataset:** A Million News Headlines (`abcnews-date-text.csv`)[cite: 3]
* **Preprocessing:** Tokenization, sequence padding, and vocabulary limiting (Top 8,000 words)[cite: 3, 8]
* **Model Architecture:** Sequential model featuring Embedding, SpatialDropout1D, stacked GRU layers, Dropout, and Dense layers with Softmax activation[cite: 3, 6]
* **Callbacks:** `EarlyStopping` and `ReduceLROnPlateau` for optimized and stable training[cite: 3, 6]
* **Generation:** Custom text generation function using temperature sampling to predict sequential words realistically[cite: 3]

---

## 🛠️ Tech Stack & Libraries
* **Python**[cite: 3]
* **TensorFlow / Keras**[cite: 3]
* **Pandas & NumPy**[cite: 3]
* **Matplotlib** (for performance visualization)[cite: 3]

---

## 🚀 Model Architecture Summary
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

## 📈 Training & Evaluation
* **Loss Function:** `sparse_categorical_crossentropy`[cite: 3]
* **Optimizer:** `Adam` (with initial learning rate of `0.001`)[cite: 3]
* Training accuracy and validation metrics are tracked via Matplotlib plots embedded in the notebook[cite: 3].

---

## 📝 Usage Example
You can test the generation function inside the notebook by providing a seed text:
```python
text = "waht are"
# The model iteratively predicts and appends subsequent words using temperature-controlled sampling.
```[cite: 3]
