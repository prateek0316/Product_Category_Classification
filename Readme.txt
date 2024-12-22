Product Category Classification
This project focuses on classifying product categories based on product descriptions. The goal is to preprocess text data, apply machine learning models, and predict product categories from given descriptions. The models are evaluated and compared based on accuracy and confusion matrix visualizations.

Table of Contents
Project Overview
Technologies Used
Data Preparation
Modeling
Evaluation
Results
How to Run
Conclusion
Project Overview
This project utilizes product descriptions to predict the category of a product. I employ text preprocessing techniques, such as lemmatization and stopword removal, to clean the data. Then, I convert the descriptions into document embeddings using pre-trained GloVe vectors, which capture the semantic meaning of the text.

Three machine learning models are trained:

Logistic Regression
Random Forest
XGBoost
The models are evaluated using accuracy scores and confusion matrices, and the best-performing model is identified.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
NLTK (Natural Language Toolkit)
TensorFlow Keras
WordCloud
Data Preparation
The data is provided in CSV format, with the following files:

train_product_data.csv: Training data with product descriptions and categories.
test_data.csv: Test data to make predictions on.
test_results.csv: Actual product categories for the test data (for model evaluation).
Preprocessing Steps
Text Cleaning:

Lowercasing all text.
Removing non-alphanumeric characters.
Tokenizing and lemmatizing the text.
Removing stopwords.
Text Representation:

GloVe embeddings are used to convert the cleaned text into document embeddings.
Label Encoding:

Product categories are encoded using LabelEncoder.
Modeling
The following models are trained and evaluated:

Logistic Regression
Random Forest
XGBoost
The models are trained using the document embeddings as features. Hyperparameters are set with default values for simplicity.

Evaluation
The models are evaluated on accuracy and confusion matrices. The accuracy scores are compared to determine which model performs best on the dataset.

Accuracy Results
Logistic Regression: 81.02%
Random Forest: 77.62%
XGBoost: 80.31%
Based on these results, Logistic Regression performs the best for this dataset.

Results
Predictions for the test data are made using the trained models. These predictions are compared with actual categories (if available) and saved to a CSV file.

Visualizations
Accuracy comparison bar plot
Confusion matrix for each model (Logistic Regression, Random Forest, XGBoost)
Word Cloud for product descriptions
How to Run
To run this project, follow these steps:

Clone the repository or download the files.
Install the necessary dependencies:
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk tensorflow wordcloud
Place the dataset files (train_product_data.csv, test_data.csv, and test_results.csv) in the appropriate directory.
Run the script:
bash
Copy code
python classify_products.py
This will train the models, evaluate them, and save the predictions to a CSV file.

Conclusion
This project demonstrates how to classify products into categories using text data and machine learning techniques. Logistic Regression performed the best among the models tested. Future improvements could involve fine-tuning the models, experimenting with different text representation techniques, or expanding the dataset.