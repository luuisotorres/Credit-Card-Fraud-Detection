<center><img src="https://www.cardrates.com/wp-content/uploads/2020/08/shutterstock_576998230.jpg"></center><br>

# Using Machine Learning to Detect Credit Card Fraud
--<br><br>
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) <br>
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)<br>

--

# Introduction

Every day, **billions** of credit card transactions are made all over the world. Considering the widespread use of smartphones and the Internet throughout the earth, more and more people are using their credit cards to make purchases online, making payments through apps,etc...<br><br>
In a scenario such as this one, it is **extremely** important that credit card companies are able to easily recognize when a transaction is a result of a fraud or a genuine purchase, avoiding that customers end up being charged for items they did not acquire.<br><br>
In this project, I used the **scikit-learn** library to develop a prediction model that is able to learn and detect when a transaction is a fraud or a genuine purchase. I tested four different classification algorithms, **Decision Tree**, **Random Forest**, **Ada Boost Classifier** and **Gradient Boosting** to identify which one of them would achieve the best results with our dataset.

# Development

In order to develop a predictive model, I used **scikit-learn** library and tested four different classification algorithms to see which one of them would achieve higher accuracy metrics for the dataset.<br><br>
Pandas, numpy, matplotlib, seaborn and plotly libs were also used to **explore**, **handle** and **visualize** relevant data for this projetc.<br><br>
The dataset used for this project was the <a href= "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud Detection</a> dataset posted on Kaggle, which contains credit card transactions that happened during the month of September, 2013 by european clients for two days.<br><br>
The dataset has the feature **time**, which shows us the seconds elapsed between each transaction and the first transaction in the dataset. The feature **amount**, containing the transaction amount and the feature **class**, which tells us if that certain transaction is **genuine or a fraud, where 1 = fraud and 0 = genuine**.<br><br>
Features V1, V2,... V28 are numerical input variables result of a <a href = "https://en.wikipedia.org/wiki/Principal_component_analysis">PCA transformation </a> whose content couldn't be displayed due to their **confidential** nature.<br><br>
During the development of this project, I've used certain tools from the **sklearn** library that would help me in achieving higher performance metrics for the models, such as the **StandardScaler**, used to alter the scale of **amount** variable and **SMOTE**, from **imblearn** library, used to deal with data imbalance. Both tools were used to avoid creating a model that would be biased towards a determined variable or a determined class.

# Evaluation Metrics for Classification Models<br>

When dealing with classification models, there are some evaluation metrics that we can use in order to see the efficiency of our models.<br><br>
One of those evaluation metrics is the **confusion matrix** which is a summary of predicted results compared to the actual values of our dataset. This is what a **confusion matrix** looks like for a binary classification problem:<br>
<center><img src= "https://miro.medium.com/max/1400/1*hbFaAWGBfFzlPys1TeSJuQ.png"></center><br><br>

**TP** is for **True Positive** and it shows the correct predictions of a model for a positive class.<br> 
**FP** is for **False Positive** and it shows the incorrect predictions of a model for a positive class.<br> 
**FN** is for **False Negative** and it shows the incorrect predictions of a model for a negative class.<br> 
**TN** is for **True Negative** and it shows the correct predictions of a model for a negative class.<br><br> 
Beyond the confusion matrix, we also have some other relevant metrics. They are:<br><br>  

### Accuracy <br><br> 
Accuracy simply tells us the proportion of correct predictions. This is how we calculate it:<br>
<center><img src = "https://www.mydatamodels.com/wp-content/uploads/2020/10/2.-Accuracy-formula-machine-learning-algorithms.png"></center><br>

### Precision <br><br> 
Precision tells us how frequently our model correctly predicts positives. This is how we calculate it:<br>
<center><img src = "https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png"></center><br>

### Recall <br><br> 
Recall, which can also be referred to as *sensitivity*, can tell us how well our model predicts the class that we want to predict. This is how we calculate it:<br>
<center><img src = "https://www.mydatamodels.com/wp-content/uploads/2020/10/3.-Sensitivity-formula.png"></center><br>

### F1 Score <br><br> 
Lastly, F1 Score is the harmonic mean of precision and recall.This is how we calculate it:<br>
<center><img src = "https://www.mydatamodels.com/wp-content/uploads/2020/10/6.-Precision-recall-1024x277.png"></center><br><br>

# Conclusion 
When we work with a **machine learning model**, we must always **know** for a fact **what it is that we're trying to get from that model**.<br><br>

In this project, our goal is to **detect fraudulent transactions when they occur**, and the model who best performed that task was the **Ada Boost Classifier** with a recall of 91.87%, correctly detecting 147 fraudulent transactions out of 160. However, it is also important to note that the Ada Boost classifier had the biggest number of false positives, that is, **1321 genuine transactions were mistakenly labeled as fraud, that's 1.54% of all genuine transactions**.<br><br>

A genuine purchase being incorrectly identified as a fraud could be a problem.<br><br>

In this scenario it is necessary to understand the business and make a few questions such as:<br><br>


- how cheap would a false positive be?<br><br>

- Would we keep the Ada Boost Classifier with the best performance in detecting frauds, while also detecting a lot of false positives or should we use the Random Forest Classifier, who also performed pretty well identified frauds (82.50% recall) and reduced the number of false positives (0.02% of genuine transactions flagged as fraud). But that would also imply in a larger number of fraudsters getting away with it and customers being mistakenly charged...<br><br>

These questions and a deeper understading of how the business works and how we want to approach solving a problem using machine learning are fundamental for a decision-making process to choose whether or not if we're willing to deal with a larger number of false positives to detect the largest amount of frauds as possible.


----

# Kaggle

I've also uploaded this *notebook* to Kaggle, where plotly graphics are interactive. If you wish to see it, please <a href = "https://www.kaggle.com/code/lusfernandotorres/91-87-recall-with-ada-boost-cc-fraud-detection/notebook">click here</a>.

# Author
**Luís Fernando Torres**
