<b align="center">Multiple Disease Prediction using ML (SVM & RF)</b>

Overview
This project implements a predictive system for diseases based on a machine learning model trained on symptoms. Using Support Vector Machines (SVM) and Random Forest (RF) classifiers, it aims to provide preliminary diagnoses for diseases based on a range of user-provided symptoms. This approach is not a substitute for professional medical advice but can aid initial screening.

Features
Symptom-Based Disease Prediction: Predicts possible diseases based on symptoms.
Machine Learning Techniques: Utilizes SVM and Random Forest for disease prediction.
Pre-Trained Models: Pre-trained models stored in the repository for quick deployment.
Interactive Interface: Runs through a Python script for easy symptom input and predictions.
Project Structure
app.py: Core application script for running disease predictions via a terminal or command line.
dispred.ipynb: Jupyter notebook with data analysis, model training, and evaluation, providing step-by-step explanations.
models/: Directory containing trained models (RandomForest8020.pkl and svm6040.pkl) for instant deployment.
images/: Contains accuracy scores and model visualizations, offering insight into model performance and structure.
Dataset
The project dataset includes records of symptoms and their associated disease labels, structured to feed directly into ML algorithms. Each record consists of multiple symptoms mapped to a specific disease, which allows the model to recognize patterns in symptoms associated with various conditions.

Getting Started
Prerequisites
Python Version: 3.7 or higher
Required Packages: Install using requirements.txt (e.g., Scikit-learn, Pandas, NumPy)
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/venkatesh-hyper/Multiple_Disease_Prediction_using_ML_SVM-RF.git
cd Multiple_Disease_Prediction_using_ML_SVM-RF
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
To run the application:

Execute app.py to start the interactive interface:
bash
Copy code
python app.py
Enter symptoms as prompted, and the model will return the most likely diseases based on input.
For detailed analysis or modifications, open dispred.ipynb in Jupyter Notebook.

Model Details
1. Support Vector Machine (SVM)
Objective: Separates classes by maximizing the margin between different disease classes.
Advantages: Effective in high-dimensional spaces; works well for symptom-based classifications.
2. Random Forest (RF)
Objective: Utilizes multiple decision trees to make more robust, accurate predictions.
Advantages: Reduces overfitting; captures nonlinear relationships.
Training Process
Data Preprocessing: Data cleaning, handling missing values, and encoding categorical features.
Feature Engineering: Selecting and transforming symptom features that improve prediction accuracy.
Model Training: SVM and RF models are trained on an 80-20 train-test split, using cross-validation to tune hyperparameters.
Evaluation Metrics
Accuracy: Measures the percentage of correct predictions.
Precision & Recall: Evaluates model reliability, especially for diseases with varying symptom overlaps.
Confusion Matrix: Visual representation of true vs. predicted disease classifications, helping analyze model strengths and weaknesses.
Evaluation metrics are visualized in dispred.ipynb, comparing SVM and RF model performances to identify which provides higher reliability for specific diseases.

Future Improvements
Expanded Dataset: Integrate more symptoms and rare diseases for broader application.
Model Tuning: Experiment with additional algorithms like Gradient Boosting for enhanced accuracy.
User Interface: Develop a graphical user interface (GUI) for better usability.
Contributing
Contributions are encouraged! To propose a change, fork the repository, create a new branch, and submit a pull request. Contributions in dataset expansion, model improvement, and interface enhancements are especially welcome.

License
This project is licensed under the MIT License. Please see the LICENSE file for more information.

Contact
For questions, feel free to reach out via LinkedIn.

