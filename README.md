# 📧 Basic Spam Email Classification
 ## 📊 Dataset
 Source: Kaggle – Spam Mails Dataset(https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
 Total samples: ~5,100 emails
 Labels: Ham (not spam) and Spam
 
 ## ⚙️ Approach
 Preprocessing: 
  Loaded dataset with pandas.
  Converted text into numerical features using CountVectorizer.
 
 Models used: 
  Support Vector Machine (SVM)
  Compared performance with Naive Bayes (MultinomialNB).

 Evaluation: 
  Split data into 80% training / 20% testing.
  Reported accuracy scores.

Optimized hyperparameters using GridSearchCV for better performance.

## 📈 Results
 Baseline (Default Scikit-Learn SVM)
<img width="770" height="391" alt="image" src="https://github.com/user-attachments/assets/e2136a32-c517-4332-ac5f-828537231199" />

 Optimized (Grid Search Cross Validation)
<img width="754" height="447" alt="image" src="https://github.com/user-attachments/assets/9f2df343-0352-4440-ba8d-21a96752f91b" />
