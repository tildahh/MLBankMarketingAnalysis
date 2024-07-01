# Machine Learning Classification: Bank Marketing Campaign Analysis

We evaluated four different classification models on a dataset from a Portuguese bank marketing campaign to predict the success of marketing efforts. The models used were Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM). Each model was optimized for the best parameters, and their performance was measured based on train and test accuracy, precision, and computational time.

### Business Problem

The primary business problem is to improve the efficiency of the bank's marketing campaigns by accurately predicting which customers will subscribe to a term deposit. This will enable the bank to:

- **Increase campaign efficiency**: By targeting the right customers, the bank can allocate resources more effectively.
- **Reduce costs**: Fewer calls will be needed to achieve the same number of successful subscriptions.
- **Improve customer experience**: By reducing unnecessary calls, customer satisfaction can be enhanced.

### The Data 
The data can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). The marketing campaigns were based on phone calls and we will build a model that predict how likely clients will subscribe to a bank term deposit. More information on the original study can be found [here.](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/CRISP-DM-BANK.pdf)

### Findings 
| Model               | Train Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision |
| ------------------- | ---------- | -------------- | ------------- | --------------- | -------------- |
| Logistic Regression | 0.045924   | 0.896080       | 0.899719      | 0.689189        | 0.722408       |
| KNN                 | 0.003624   | 0.900928       | 0.893085      | 0.672497        | 0.593103       |
| Decision Tree       | 0.011317   | 0.902906       | 0.893085      | 0.713612        | 0.602532       |
| SVM                 | 50.877737  | 0.894963       | 0.898061      | 0.695035        | 0.727273       |

* The best performance metrics of these models will be Precision since it's a highly imbalanced dataset and it will give us the best result for the customers that will subscribe. 

* KNN had the lowest performance among the models, indicating potential overfitting and lower precision, making it less reliable for this dataset.

* **SVM demonstrated the highest performance in terms of accuracy and precision but required significantly more computational time for training.**

* Logistic Regression provided a good balance of performance and efficiency, demonstrating high accuracy and precision with a relatively quick training time. 

* **Winners:** SVM would be the best model to use if computational time is not a concern, but Logistic Regression would be a more practical choice for a balance of performance and efficiency.

#### Model Performance Comparison
![Model Accuracy](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/images/final_accuracy_comparison.png)

![Model Precision](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/images/final_precision_comparison.png)

![Model Percision Comparison](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/images/models_precision_comparison.png)
* This graph shows the models ability to predict bank customers that will subscribe to a term deposit. 
* SVM has the highest test Precision after hyperparameter tuning. 
* All models except of Logistic Regression shows signs of overfitting before hyperparameter tuning.

#### Model Evaluation
![ROC Curves](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/images/roc_curves.png)

![Confusion Matrix](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/images/confusion_matrix_SVM.png)
* Confusion Matrix for our best model, SVM, shows that the model is better at predicting True Positives than False Positives.

#### Next Steps and Recommendations: 
* Feature Testing: Further test using a different number of features to see how it affects model performance.
* Hyperparameter Tuning: Continue tuning the hyperparameters to optimize the models further.
* Model Testing: Test Random Forest or XGBoost models instead of Decision Tree since they are highly prone to overfitting and can potentially offer better performance.

This approach ensures you are making data-driven decisions, optimizing model performance, and considering practical constraints like computational time and overfitting tendencies.

For a detailed analysis, refer to the [Jupyter Notebook](https://github.com/tildahh/MLBankMarketingAnalysis/blob/main/prompt_III.ipynb).
