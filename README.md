# 🧠 AI Story Generator (RNN-based Text Generation)

This project is a **story generation model** built using **Recurrent Neural Networks (RNNs)** trained on the complete collection of **Sherlock Holmes stories** by Sir Arthur Conan Doyle.  
The model learns writing patterns, vocabulary, and sentence structures from the text and generates new, Sherlock-style stories word by word.

---

## 🚀 Features
- Generates **coherent story text** using a trained RNN model.  
- Implements **character-level text generation** for creative writing.  
- Preprocessing includes:
  - Lowercasing and cleaning text  
  - Removing punctuation and spaces  
  - Character-to-integer encoding  
- Trained on **Sherlock Holmes stories dataset** (public domain).  
- Adjustable generation **length** and **temperature** to control creativity.

---

## 🧩 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**

---

## ⚙️ How It Works
1. The Sherlock Holmes text is cleaned and converted into lowercase.  
2. Each unique character is assigned an integer ID.  
3. Sequences of fixed length are created to train the model to predict the next character.  
4. A **Recurrent Neural Network (RNN)** learns these sequences using an LSTM or GRU layer.  
5. Once trained, the model generates new text by predicting the next character step-by-step.

---

## 🧪 Example Output
> “Holmes looked at me thoughtfully. ‘Watson,’ he said,  
> ‘there is more to this case than meets the eye.’”

---

## 📊 Training Details
- **Dataset:** All Sherlock Holmes stories (public domain text)  
- **Model:** RNN / LSTM with dense output layer  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Character-level accuracy  

---

## 💡 Future Improvements
- Add **Transformer-based text generation** for better context understanding  
- Create a **web interface** to generate stories interactively  
- Add **prompt-based generation** where users can continue custom stories  

---

## 📁 Project Structure
```
AI-Story-Generator/
│
├── data/
│   └── sherlock_stories.txt
│
├── notebooks/
│   └── model_training.ipynb
│
├── model/
│   └── story_generator.h5
│
├── generate.py
├── preprocess.py
└── README.md
```

---

## 🧑‍💻 Author
**Subham Mondal**  
Final Year B.Tech (Computer Science)  
Interested in Machine Learning and Artificial Intelligence  
[GitHub](https://github.com/) • [LinkedIn](https://linkedin.com/in/)

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).

---

⭐ *If you like this project, consider giving it a star on GitHub!*