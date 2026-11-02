# Student Outcome Classifier

## a. Problem statement

Higher-education institutions want to spot students at risk of dropping out early enough to help them. The task is a **3-class classification** problem: using information available at enrolment plus first-year academic results, predict whether a student will end up as a **Dropout**, still **Enrolled**, or a **Graduate** at the end of the normal course duration.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, compared on six metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset id 697 — https://archive.ics.uci.edu/dataset/697 |
| DOI / Licence | 10.24432/C5MC89 · CC BY 4.0 |
| Instances | 4,424 (one row per student) |
| Features | 36 (integer, real and integer-coded categorical) |
| Target | `Target` — Graduate (2,209), Dropout (1,421), Enrolled (794) |
| Missing values | None |

The data comes from a higher-education institution and covers students on undergraduate degrees such as agronomy, design, education, nursing, journalism, management, social service and technologies. Features fall into four groups:

- **Demographic** — marital status, nationality, gender, age at enrolment, displaced, international
- **Socio-economic** — parents' qualifications and occupations, scholarship holder, debtor, tuition fees up to date, special educational needs
- **Academic path at enrolment** — application mode/order, course, day/evening attendance, previous qualification and grade, admission grade
- **Academic performance** — curricular units credited / enrolled / evaluated / approved / grade / without evaluation for the 1st and 2nd semester
- **Macro-economic** — unemployment rate, inflation rate, GDP

Note that the classes are imbalanced: "Enrolled" is only 18% of the data.

**Preprocessing:** column names are stripped of stray whitespace (the raw file has a trailing tab in one header). The data is split 80/20 with stratification on `Target` (`random_state=42`), giving 3,539 training rows and 885 test rows. The 885-row test split is saved as `test_data.csv`. `StandardScaler` is fitted on training data only, inside each model's pipeline (not used for the Decision Tree).

## c. GitHub repository

Repository link: **https://github.com/2026aj05080Shivam/ml-classification-app**

Live Streamlit app: **https://ml-classification-2026aj05080.streamlit.app**

```
ml-classification-app/
│-- app.py               # Streamlit app (loads saved models, never retrains)
│-- train.py             # trains the 5 models, writes model/ and test_data.csv
│-- metrics.py           # shared evaluation-metric function
│-- requirements.txt
│-- README.md
│-- test_data.csv        # held-out 20% test split (features + Target)
│-- data/data.csv        # original UCI dataset
│-- model/
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- support_vector_machine.joblib
│   │-- metadata.json    # feature list, classes, split sizes
│   │-- metrics.csv      # metrics table below
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler → LogisticRegression(max_iter=2000) |
| Decision Tree | DecisionTreeClassifier(max_depth=8, min_samples_leaf=10) |
| KNN | StandardScaler → KNeighborsClassifier(n_neighbors=15, weights="distance") |
| Naive Bayes | StandardScaler → GaussianNB() |
| Support Vector Machine | StandardScaler → SVC(kernel="rbf", C=1.0, probability=True) |

**Metric definitions (multi-class):** Precision, Recall and F1 are **weighted** averages across the three classes; AUC is **one-vs-rest, macro-averaged** using predicted probabilities; MCC is the multi-class Matthews correlation coefficient.

### Comparison table (test set, 885 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8778 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7198 | 0.8450 | 0.7153 | 0.7198 | 0.7167 | 0.5403 |
| KNN | 0.7017 | 0.8161 | 0.6852 | 0.7017 | 0.6721 | 0.5009 |
| Naive Bayes | 0.6588 | 0.7893 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine | 0.7582 | 0.8664 | 0.7465 | 0.7582 | 0.7454 | 0.5986 |

### Observations

Per-class recall on the test set is quoted below as Dropout / Enrolled / Graduate.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (Accuracy 0.768, AUC 0.878, MCC 0.615). Strong on Graduate (recall 0.925) and Dropout (0.768) but weak on the minority Enrolled class (0.333). The features — especially semester units approved and grades — relate to the outcome in a largely linear way after scaling, so a simple linear model does well. |
| Decision Tree | Mid-table (Accuracy 0.720, MCC 0.540). It has the **highest Enrolled recall of all models (0.409)**, so it is less biased toward the majority class, but its lower Dropout (0.676) and Graduate (0.860) recall pull the overall score down. Depth was capped at 8 with min 10 samples per leaf to limit overfitting. |
| KNN | Accuracy 0.702 but the lowest F1 apart from Naive Bayes (0.672). It favours the majority class heavily: Graduate recall 0.937 versus Enrolled 0.195. With 36 dimensions, many of them integer-coded categories, distance between students is less meaningful, which hurts KNN. |
| Naive Bayes | Weakest model on all metrics (Accuracy 0.659, MCC 0.428). The Gaussian, feature-independence assumption does not fit this data: many features are coded categories rather than continuous values, and the 1st- and 2nd-semester features are strongly correlated with each other (e.g. units enrolled r = 0.94, units approved r = 0.90). Enrolled recall is only 0.182. |
| Support Vector Machine | A close second (Accuracy 0.758, AUC 0.866, MCC 0.599). It has the highest Dropout precision of all models (0.841), meaning a "Dropout" prediction from it is the most reliable, though Dropout recall (0.725) is lower than Logistic Regression's. The RBF kernel adds no clear gain over the linear model here. |

**Overall winner: Logistic Regression.** It has the best accuracy, AUC, F1 and MCC, and it is also the fastest and easiest to interpret. SVM is a close second. All models find the **Enrolled** class hardest; it is the smallest class and lies between the other two outcomes. If catching every at-risk student matters more than overall accuracy, class weighting or resampling would be the next thing to try.

## e. Streamlit app

Features:
1. **CSV upload** (sidebar) — upload `test_data.csv` or any file with the same 36 feature columns plus `Target`. A sample file can be downloaded from the sidebar.
2. **Model selection dropdown** — choose any of the 5 saved models.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
4. **Confusion matrix and classification report** for the chosen model, plus row-by-row predictions that can be downloaded.
5. **Compare all models** tab — metric table for all 5 models on the uploaded data, with the best value in each column highlighted, and a bar chart for any metric.

The app only loads the `.joblib` files and calls `predict` / `predict_proba`. It never retrains.

## f. How to run locally

```bash
pip install -r requirements.txt
python train.py          # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

`requirements.txt` pins `scikit-learn==1.9.1` because the saved models must be loaded with the same scikit-learn version that created them.
