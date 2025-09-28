## 🚀 Machine Learning Algorithms in Java

This repository contains Java implementations of classic machine learning algorithms, designed for classification and evaluation tasks such as handwritten digit recognition.

## 📌 Implemented Algorithms

🔹 K-Nearest Neighbors (KNN) – supervised classification using Euclidean distance

🎲 Random Guess – simple baseline classifier for comparison

🧠 Self-Organizing Map (SOM) – unsupervised learning inspired by biological neural networks

🌳 C4.5 Decision Tree – classification with information gain and pruning

## ✨ Features

✔️ Two-fold cross-validation (train/test swapping)  
✔️ Confusion matrix output for each algorithm  
✔️ Accuracy evaluation and comparison  
✔️ Modular and extendable code structure  

## 📂 Project Structure
/src/machine_learning_algorithms/  

    AlgorithmRunner.java         # Main runner class  
    C45_Algorithm.java           # C4.5 Decision Tree  
    KNN_Algorithm.java           # K-Nearest Neighbors  
    Random_Guess_Algorithm.java  # Random baseline  
    SOM_Algorithm.java           # Self-Organizing Map  

## ⚙️ Requirements

☕ Java 8+

📑 CSV dataset files (e.g., UCI Optical Recognition of Handwritten Digits)

## ▶️ How to Run

#### 1. Clone the repository

git clone https://github.com/your-username/machine-learning-algorithms.git  

cd machine-learning-algorithms

#### 2. Place your dataset files in the appropriate folder.

#### 3. Run the main program

- Open AlgorithmRunner.java

- Compile & run to test algorithms on the dataset

## 📊 Example Results (Two-Fold Test)

KNN → ~98% accuracy 

SOM → ~85% accuracy

C4.5 → ~55% accuracy

Random Guess → ~10% accuracy (baseline)

## 📖 License

📝 This project is provided for learning and educational purposes.  

Feel free to **fork** and **extend** with more algorithms!
