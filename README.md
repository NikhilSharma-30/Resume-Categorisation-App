# Resume-Categorisation-App
Resume Categorisation App: This project is aimed at automating the categorization of resumes into predefined job categories. The goal is to use machine learning techniques to classify resumes based on their content into various categories such as software engineering, marketing, finance, etc.

Project Overview With the increasing number of resumes being submitted for job openings, it becomes challenging to manually categorize and prioritize them. This machine learning-based solution helps recruiters automate the sorting process by categorizing resumes based on key skills, experiences, and keywords.

Key Technologies: Python 3.x, Scikit-learn, Pandas, Natural Language Processing (NLP), TF-IDF (Term Frequency-Inverse Document Frequency), KNN / Random Forest Classifier / Support Vector Machines (SVM)

Dataset The dataset consists of resumes, each labeled with its corresponding job category (e.g., Software Engineer, Data Scientist, Marketing Specialist). You can use any publicly available resume dataset or create your own by manually labeling resumes into different categories.

Features: Preprocessing of Text Data: Clean the resume text by removing stop words, special characters, and normalizing case. Feature Extraction: Utilize TF-IDF or word embeddings to convert the text into a machine-readable format. Model Training: Train various classifiers like Logistic Regression, Random Forest, and SVM to categorize resumes into job categories. Evaluation: Evaluate the model’s performance using metrics like accuracy, precision, recall, and F1-score.

Setup Instructions Prerequisites, Python 3.x, Scikit-learn, Pandas, Numpy, Matplotlib (for visualizations), Jupyter Notebook or any Python IDE

Model Architecture This project uses traditional machine learning methods for text classification: Text Preprocessing: Tokenization, stopword removal, stemming/lemmatization. Feature Extraction: TF-IDF vectorization is used to convert resumes into numerical data.

Classification Models: KNN, Random Forest Classifier, Support Vector Machine (SVM). The best model is selected based on evaluation metrics such as accuracy and F1-score.

How to Use Preprocess the Resume Data: Load the resumes and clean the text data by removing unnecessary information. Extract Features: Convert the resume text into numerical form using techniques like TF-IDF. Train the Model: Use a classification algorithm (e.g., Logistic Regression, Random Forest) to train the model. Predict Categories: Once trained, you can predict the job category for new resumes.

Valuation Metrics: The model is evaluated using the following metrics: Accuracy: Proportion of correct predictions. Precision: Proportion of positive predictions that are actually correct. Recall: Proportion of actual positive cases that are correctly identified. F1-Score: Harmonic mean of precision and recall
