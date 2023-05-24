
# Phase 3 Project - Choosing a Dataset

Dennis Mwanzia
Phase 3 Project
Classification of Water Wells in Tanzania


# Table of Contents
1.	Introduction
2.	Aim
3.	Objectives
4.	Data Source
5.	Client
6.	Import Libraries, Modules & Functions
7.	Obtain Data/Load Data
8.	Merging the Data Frames
9.	Data Cleaning
10.	Column description and Understanding
11.	Data Understanding
12.	Checking for null values
13.	Inspecting and dealing with columns with Missing data
14.	Dealing with columns that have similar data
15.	The Target Column Inspection
16.	Describing the numerical columns
17.	Checking Other Irrelevant Columns & Initial Feature Engineering
18.	Visualization of Target vs Selected Features
19.	Visualizations using GeoPandas
20.	Importing Tanzania Geographical data
21.	Mapping Waterpoints by Function
22.	Well Functionality by Management
23.	Mapping the status of wells by Level of water quantity
24.	Waterpoint functional status by Basin
25.	Mapping Waterpoint Type Map
26.	Mapping Well sources
27.	Waterpoint Payment Type Mapping
28.	Visualization Other Categorical columns
29.	Modelling
a)	Initial Model/Dummy Model
b)	Decision Tree Classifier
c)	Logistic Regression Model
d)	Light Gradient Boosting Machine (LGBM)
e)	XGBClassifier
f)	Random Forest
g)	Random Forest with SMOTE

30.	Interpretations
	Conclusion
	Recommendation

## Introduction
Tanzania, is an african developing country that is struggling with providing clean water to its population of over 63 million people as at 2021. There are many waterpoints already established in the country, but some are in need of repair while others have failed altogether. The main aim of this project is to build a model classifier that can predict the condition of water well using information about the type of pump, when it was installed, when the water well was constructed, how it is operated etc.
Understanding which waterpoints will fail can help to improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.


## Aim
The goal of this project is to develop a model that predicts the functionality of water points which authorities in Tanzania can use to understand which water points are functional, nonfunctional, and functional but it needs to be repaired. The model will help the government of Tanzania to find the maintainance needs of wells while also being able to predict future well needs. The government will be able to prioritize the distribution of resources to optimize availability of water to communities throughout the country.



## Objectives
1. Develop a classifier to classify the status of water wells in Tanzania
2. Develop recommendations based on the status of the functional water wells in Tanzania.


## Data
The original data was obtained from the DrivenData 'Pump it Up: Data Mining the Water Table' competition which is data aggregrated from Taarifa waterpoints dashboard which aggregrates data from the Tanzania Ministry of Water. Taarifa is an open source platform for the crowd sourced reporting and triaging of infrastructure related issues. There were four different datasets namely;
1. submission format
2. training set
3. test set 
4. train labels set which contains status of wells. 
## Data Cleaning & Dropping Features


## Initial Model/Dummy Model

The confusion matrix and classification report for the Dummy Model indicate that the model is not predicting any positive instances (classes 1 and 2). This suggests that the model is not learning from the data and is predicting the majority class (class 0) for all instances.

The confusion matrix shows that the model predicts all instances as class 0, resulting in true positives (25802) for class 0 and no predictions for classes 1 and 2. The recall for class 0 is 1.00, which means that the model correctly identifies all instances of class 0. However, the recall for classes 1 and 2 is 0.00, indicating that the model fails to identify any instances of these classes.

The classification report further confirms these observations. The precision, recall, and F1-score for classes 1 and 2 are all 0.00, indicating that the model does not make any positive predictions for these classes. The accuracy of the model is 0.54, which is relatively high due to the majority class being correctly predicted, but it does not reflect the model's actual performance.

Therefore, the Dummy Model is not a useful model for classification as it fails to predict any positive instances and performs poorly for classes 1 and 2. Therefore, I proceeded to develop another baseline model which i wanted to build on.


## Decision Tree Classifier

Precision: Precision measures the accuracy of the positive predictions. The precision for label 0 is 0.68, which means that 68% of the samples predicted as label 0 are actually label 0. Precision for label 1 and label 2 are 0.0 and 0.75, respectively.

Recall: Recall measures the ability of the model to correctly identify the positive samples. The recall for label 0 is 0.90, indicating that 90% of the actual label 0 samples were correctly identified. Recall for label 1 is 0.0, indicating that none of the actual label 1 samples were correctly identified. Recall for label 2 is 0.54, indicating that 54% of the actual label 2 samples were correctly identified.

F1-score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. The F1-score for label 0 is 0.77, label 1 is 0.0, and label 2 is 0.63. 

Overall, the DecisionTreeClassifier model has an accuracy of 0.70, meaning that it correctly predicts the labels for 70% of the samples. The model performs well in identifying label 0, with high precision and recall. However, it struggles to identify label 1, as indicated by the low precision and recall for that class. The model shows moderate performance for label 2, with a reasonably high precision and a moderate recall.

This model performed quite well compared to the dummy model and therefore, i proceeded to improve on it. It suffered from class imbalance. I proceeded to use the Logistic Regression Model.



## Logistic Regression Model
The accuracy scores for both the training and test sets are similar, with an accuracy of approximately 71.6% for the training set and 71.5% for the test set. This implies there is no overfitting or underfitting it.

Looking at the confusion matrices, it is evident that the model struggles with predicting class 1 in both the training and test sets. Class 1 is consistently predicted as class 0 or class 2, resulting in a high number of false negatives and low recall for class 1.

Overall, the model's performance is relatively low, especially for class 1, which is not predicted correctly at all. I proceeded further to investigate other models.


## Light Gradient Boosting Machine (LGBM)

Recall: The overall recall of the model is 0.784, indicating that it correctly identifies approximately 78.4% of the instances across all classes.

Class 0 (functional): The precision is 0.76, indicating that 76% of the instances predicted as functional are actually functional. The recall is 0.92, indicating that 92% of the functional instances are correctly identified.

Class 1 (functional needs repair): The precision is 0.69, indicating that 69% of the instances predicted as needing repair are actually in need of repair. The recall is 0.18, indicating that only 18% of the instances needing repair are correctly identified. The F1-score is 0.28, which suggests that the model struggles to accurately classify instances in this class. Class 2 (non-functional): The precision is 0.85, indicating that 85% of the instances predicted as non-functional are indeed non-functional. The recall is 0.70, indicating that 70% of the non-functional instances are correctly identified.

The F1-score is 0.77, suggesting a relatively good performance in classifying non-functional instances.

Accuracy: The overall accuracy of the model is 0.78, indicating that it correctly predicts approximately 78% of the instances across all classes.

The LGBM model shows relatively good performance in correctly classifying functional and non-functional instances, but it struggles with accurately identifying instances of class 1.


## XGBClassifier

Recall: The average recall across all classes is 0.7806, indicating that the model is able to correctly identify approximately 78.06% of the positive instances in the dataset.

Classification Report: The precision, recall, and F1-score are reported for each class (0, 1, 2). Class 0 has a precision of 0.75, indicating that 75% of the predicted instances for class 0 are actually correct. Class 0 also has a recall of 0.93, indicating that 93% of the actual instances of class 0 are correctly predicted by the model.

Overall, the XGBClassifier model achieves an accuracy of 78% on the test set, with a higher precision, recall, and F1-score for class 0 compared to classes 1 and 2. The model performs relatively well in identifying instances of class 0 but struggles with the minority classes (1 and 2) due to their imbalanced nature. I will consider strategies to address class imbalance and also explore another model (LGBM) to improve performance on the minority classes.


## Random Forest

Model: Random Forest Recall: 0.9014730639730639

Precision: The precision for label 0 is 0.88, indicating that 88% of the samples predicted as label 0 are actually label 0. Precision for label 1 is 0.85, and for label 2 is 0.94. These values indicate a high level of accuracy in predicting the positive samples for each label.

Recall: The recall for label 0 is 0.96, meaning that 96% of the actual label 0 samples were correctly identified. Recall for label 1 is 0.62, indicating that 62% of the actual label 1 samples were correctly identified. Recall for label 2 is 0.87, indicating that 87% of the actual label 2 samples were correctly identified. The model performs well in identifying label 0 and label 2, but has relatively lower performance for label 1 because of class imbalance.

F1-score: The F1-score for label 0 is 0.92, label 1 is 0.72, and label 2 is 0.90.

Overall, the Random Forest model has an accuracy of 0.90, meaning that it correctly predicts the labels for 90% of the samples. The model performs well in identifying label 0 and label 2, with high precision and recall for both classes. However, it shows relatively lower performance for label 1, as indicated by the lower precision and recall for that class.

This model improves significantly from DecisionTreeClassifier.

## Random Forest with SMOTE

Accuracy:The overall accuracy is 0.87, indicating that the model correctly predicts 87% of the instances.

Based on the classification report, the model performs well in predicting class 0 and class 2, with high precision, recall, and F1-scores. However, it performs relatively poorly in predicting class 1, with lower precision, recall, and F1-score values.
The model accuracy score has actually declined when applying SMOTE hence its better to have the final model without tuning using SMOTE method.


## Interpretation, Recommendations & Conclusions
Based on the results of the Random Forest model, we can draw the following conclusions: I observed that the precision for functional waterpoints (label 0) is 0.88, implying that 88% of the samples predicted as functional are actually functional. The precision for waterpoints that need repair (label 1) is 0.85, indicating 85% accuracy in predicting this category. For non-functional waterpoints (label 2), the precision is 0.94, demonstrating 94% accuracy. These high precision values suggest that the Random Forest model performs well in accurately identifying positive samples for each label.

Moreover, the findings show that the recall for functional waterpoints (label 0) is 0.96, meaning that 96% of the actual functional samples were correctly identified. However, the recall for waterpoints that need repair (label 1) is 0.62, indicating that only 62% of the actual samples needing repair were correctly identified. The recall for non-functional waterpoints (label 2) is 0.87, indicating that 87% of the actual non-functional samples were correctly identified. While the model performs exceptionally well in identifying functional and non-functional waterpoints, it has relatively lower performance in correctly identifying waterpoints that need repair.

I found that the F1-score for functional waterpoints (label 0) is 0.92, for waterpoints needing repair (label 1) is 0.72, and for non-functional waterpoints (label 2) is 0.90. These scores indicate good overall performance, with functional and non-functional waterpoints achieving higher F1-scores compared to waterpoints that need repair.

Overall, the Random Forest model demonstrates significant improvement compared to the Initial Model/Dummy Model, Decision Tree Classifier, Logistic Regression Model, Light Gradient Boosting Machine (LGBM), XGBClassifier and Random Forest with SMOTE

It achieved an accuracy of 0.90, indicating that it correctly predicts the labels for 90% of the samples. The model performs well in accurately identifying functional and non-functional waterpoints, but it struggles to correctly identify waterpoints that need repair, possibly due to class imbalance. To further enhance the model's performance, it is recommended to address the class imbalance issue and explore techniques such as oversampling or undersampling to improve the prediction of waterpoints needing repair.

In conclusion, the Random Forest model shows promise in predicting the condition of waterpoints in Tanzania. By focusing resources on waterpoints that are likely to fail or need repair, implementing more reliable pump types, and ensuring accurate data gathering in collaboration with local governments, we can improve maintenance operations and ensure the availability of clean and potable water to communities across the country.
Using the given training set and labels set, I developed a predictive model which can be applied to the test set to determine status of the wells.

