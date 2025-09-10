# CS4680 Prompt Engineering  
**Assignment 1 – Machine Learning Exercise**

---

## 1. Problem Identification  

The goal of this machine learning exercise was to apply **classification analysis** to a real-world problem using scikit-learn.  
The chosen problem is **spam detection**: determining whether an incoming SMS message is **spam** (unwanted or fraudulent) or **ham** (legitimate).  

This problem is well-suited for machine learning classification because:  
- There was only two categories (spam or ham) that the messages fell under.
- The input feature (message text) contains natural patterns that were easy to distinguish between what was fraudulent and legitimate.
- Spam messages poses very serious risks for communication security, and the proper implementation of spam detection is paramount to keeping communications safe. 

**Target Variable**: SMS category (spam or ham)  
**Features**: SMS message text, transformed into numerical features using **TF-IDF vectorization** 

---

## 2. Data Collection  

The dataset used was the **[SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**.  

- **Total messages**: 5,572  
- **Label distribution**:  
  - Ham (legitimate): 4,825  
  - Spam: 747  

---

## 3. Model Development  

Two classification models from scikit-learn were chosen for comparison:  

1. **Logistic Regression**  
   - A linear model that estimates probabilities of belonging to a class.  
   - Serves as a strong baseline for binary classification tasks.  

2. **Multinomial Naive Bayes**  
   - A probabilistic classifier that assumes independence among features.  
   - Particularly well-suited for **text classification** problems like spam detection.  

**Preprocessing Steps**:  
- Converted SMS messages into numeric vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.  
- Applied **train/test split** (80% training, 20% testing) with stratification to preserve spam/ham ratios.  

---

## 4. Model Evaluation  

Both models were trained and tested on the dataset. Their performance was compared using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrices**.

### Logistic Regression Results
- **Accuracy:** 96.7%  
- **Ham:** Precision = 0.96, Recall = 1.00, F1 = 0.98  
- **Spam:** Precision = 1.00, Recall = 0.75, F1 = 0.86  

**Interpretation**: Logistic Regression correctly classified almost all legitimate messages and avoided false positives, but it missed ~25% of spam messages.  

---

### Multinomial Naive Bayes Results
- **Accuracy:** 97.0%  
- **Ham:** Precision = 0.97, Recall = 1.00, F1 = 0.98  
- **Spam:** Precision = 1.00, Recall = 0.77, F1 = 0.87  

**Interpretation**: Naive Bayes slightly outperformed Logistic Regression. Like Logistic Regression, it never mislabeled ham as spam (no false positives), but it achieved slightly better recall for spam detection.  

---

### Confusion Matrix (Naive Bayes)
| True Label | Predicted Ham | Predicted Spam |
|------------|---------------|----------------|
| Ham        | 966           | 0              |
| Spam       | 34            | 115            |

**Observation**: The model never classified legitimate messages as spam, but 34 spam messages were misclassified as legitimate.  

---

## 5. Discussion  

Both Logistic Regression and Naive Bayes achieved **high overall accuracy (~97%)**, showing that SMS spam detection is effectively addressed using classification analysis.  

- **Strengths:**  
  - Very high precision on spam (1.00) → ensures no legitimate messages are wrongly flagged.  
  - High recall on ham (1.00) → correctly identified nearly all legitimate messages.  
- **Weaknesses:**  
  - Moderate recall on spam (~0.75–0.77) → some spam messages still slip through.  
  - This trade-off is common: prioritizing precision reduces false alarms but can lower recall.  

In real-world spam filtering, **precision is often prioritized over recall** as it is more acceptable to miss a spam message than to wrongly flag an important legitimate one, which
can hurt the user experience.   

---

## 6. Conclusion  

In conclusion,
Both models proved to be extremely accurate and precise in detecting which messages were fraudulent/unwanted. However, the Multinonial Naive Bayes model slightly outperformed the latter
in ensuring no legitimate messages were flagged as spam. 
---
