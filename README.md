# 🏷️ Hidden Markov Model POS Tagger

A statistical Part-of-Speech (POS) tagger implemented using a **Hidden Markov Model (HMM)** and the **Viterbi algorithm** for sequence decoding.

---

## 📋 Project Overview

This project builds a **POS tagger** trained on the Wall Street Journal (WSJ) portion of the Penn Treebank. It predicts the most likely sequence of POS tags for a given sentence using:

- Hidden Markov Models (HMM)
- Log-space probability calculations
- Custom handling of out-of-vocabulary (OOV) words

---

## ✨ Features

- **Hidden Markov Model** with emission and transition probabilities
- **Viterbi decoding** in log-space for optimal tag sequence
- **Add-one smoothing** for unseen transitions
- **OOV Word Handling** using feature-based tagging
- **No external libraries** — built using Python standard library only

---

## 🔧 Technical Implementation

### Data Processing

- Parses training data (`WSJ_02-21.pos`) to extract word-tag pairs
- Computes:
  - **Tag transition probabilities**
  - **Word emission probabilities**
- Applies **add-one smoothing** for robust modeling

### Viterbi Algorithm (Log-Space)

Efficiently computes the most probable tag path:
```

At each word:
For each tag:
max\_prob = max(
previous\_prob\[tag\_prev]
\+ log(transition\[tag\_prev]\[tag])
\+ log(emission\[tag]\[word])
)
Track backpointers

````

### OOV Handling

- Classifies unseen words based on:
  - Numeric patterns → `<NUM>`
  - Capitalization → `<CAP>`
  - Plural suffix → `<PLURAL>`
  - Punctuation → `<PUNCT>`
  - Unknown → `<UNK>`

- Distributes probabilities using rare word tag frequencies.

---

## 📚 Dataset

You’ll need the following files (from Penn Treebank WSJ):
- `WSJ_02-21.pos` → Training set (~950k words)
- `WSJ_24.pos` → Development set
- `WSJ_23.words` → Test set (untagged)

All must be placed in the same directory as the script.

---

## 🛠️ Requirements

- **Python 3.x**
- No external dependencies

---

## 🚀 How to Run

```bash
python word_tagger.py
````

Output will be saved to:

```
submission.pos
```

Each line in the output will follow:

```
word/tag
```

---

## 📊 Performance

| Metric                 | Value |
| ---------------------- | ----- |
| Overall Accuracy       | \~96% |
| In-Vocabulary Accuracy | \~97% |
| OOV Word Accuracy      | \~85% |

---

## 🌟 Technical Highlights

* Transition and emission matrices built from scratch
* Efficient Viterbi decoding in log space
* Clean, interpretable output
* Sophisticated OOV classification based on word features

---

## 🔮 Future Improvements

* Integrate Good-Turing or Kneser-Ney smoothing
* Extend to other languages or multilingual corpora
* Add a GUI or web demo using Flask or Streamlit
* Incorporate contextualized embeddings for OOV words (e.g., BERT)

---

## 📝 License

This project is released under the **MIT License**. See the `LICENSE` file for more details.
