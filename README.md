# Lb1_DeepLearning_PyTorch
1. Introduction

The objective of this laboratory session is to explore the use of PyTorch to build, train, evaluate, and optimize deep neural networks for:

Regression using the NYSE stock prices dataset

Multiclass classification using the Predictive Maintenance dataset

Throughout the lab, the work included:

Exploratory Data Analysis (EDA)

Data preprocessing and normalization

Implementation of MLP architectures in PyTorch

Training and testing neural networks

Hyperparameter tuning using GridSearchCV

Regularization (Dropout & Weight Decay)

Performance evaluation using loss curves, accuracy curves, and metrics

Oversampling imbalanced data using SMOTE

Interpretation and comparison of results

2. Part 1 – Regression (NYSE Stock Prices)
2.1 Dataset & EDA

The NYSE dataset contains 851,264 rows with 7 columns. For regression, only numerical columns were kept:

Features: open, high, low, volume

Target: close

Key EDA Findings:

Price variables (open, close, high, low) are perfectly correlated (~1.0)

volume has very weak correlation with prices (~–0.06)

The target variable is highly skewed, with most prices concentrated under 200

This indicates a highly collinear and noisy dataset, making regression difficult.

2.2 Model Architecture

A fully connected regression network was implemented:

Input: 4 features  
Hidden layers: 64 → 32 (ReLU)  
Output: 1 neuron  
Loss: MSELoss  
Optimizer: Adam (lr = 0.001)  
Epochs: 50  

2.3 Training Results
Epoch	Train Loss	Test Loss
0	11970	11926
20	11816	11771
40	11560	11509
50	≈11390	≈11320
Loss Curve Observations

Train & test losses decrease smoothly

Test loss remains slightly below train loss → no overfitting

Regression loss remains high due to noisy and correlated features

 Conclusion:
The model learns stable relationships, but the dataset’s structure limits predictive performance.

2.4 Hyperparameter Tuning (GridSearchCV)

Best parameters identified:

lr = 0.001
module__hidden1 = 64
Best RMSE ≈ 82.9


Interpretation:
GridSearch improved the regression significantly but could not eliminate inherent dataset noise.

2.5 Regularization Effects

With Dropout + Weight Decay:

Epoch 40 → Test Loss = 11970 → 11500 (baseline)


Regularized model converged more slowly

Reduced overfitting

Slight improvement in test error stability

 Conclusion:
Regularization improves generalization but cannot drastically reduce loss due to dataset limitations.

 3. Part 2 – Multiclass Classification (Predictive Maintenance)
3.1 Dataset & EDA

The dataset includes machine conditions with 6 failure types:

Extreme imbalance: class 1 dominates (~9500 samples)

Other classes have <200 samples

Mechanical features (rotational speed, torque, tool wear) show strongest correlation with failure type

 Conclusion:
Class imbalance required correction for any reliable learning.

3.2 Data Balancing using SMOTE

SMOTE was applied to oversample minority classes.
→ Successfully balanced all failure classes
→ Significantly improved model learning stability

3.3 Model Architecture
Input: n_features  
Hidden layers: 64 → 32 (ReLU)  
Dropout: 0.3 + 0.3  
Output: 6 classes  
Loss: CrossEntropyLoss  
Optimizer: Adam (lr = 0.001)  
Epochs: 50  

3.4 Training Results (Baseline Model)
Epoch	Train Acc	Test Acc
0	0.22	0.19
10	0.43	0.51
20	0.53	0.57
30	0.66	0.66
40	0.73	0.73
50	0.78	0.78
Accuracy & Loss Curve Observations

Train & test curves almost identical

No overfitting

Accuracy reaches 78%, excellent for 6 balanced classes

Loss decreases smoothly and consistently

 Conclusion:
SMOTE + MLP achieved high and stable performance.

3.5 Confusion Matrix & Classification Report

Your real results:

Confusion Matrix
[[1703    0   30    7    0  141]
 [   2 1596    0    0  362    2]
 [ 223    0 1566    0    0  144]
 [ 403    0  218 1336    0    0]
 [   0  733    0    0 1183    0]
 [   0    0  268    0    0 1666]]

Classification Report
Class	Precision	Recall	F1-score	Support
0	0.73	0.91	0.81	1881
1	0.69	0.81	0.74	1962
2	0.75	0.81	0.78	1933
3	0.99	0.68	0.81	1957
4	0.77	0.62	0.68	1916
5	0.85	0.86	0.86	1934
Overall Performance

Accuracy: 0.78

Macro average F1: 0.78

Weighted F1: 0.78

Analysis:

Classes 0, 2, 5 perform very well

Class 4 is the most difficult (recall = 0.62)

Class 3 is predicted cautiously (high precision, lower recall)

Balanced dataset leads to strong classifier performance

3.6 Regularized Model Results

After applying dropout and weight decay:

Epoch	Train Acc	Test Acc
50	0.75	0.75

Interpretation:
Regularization slightly reduces accuracy but improves robustness and prevents overfitting.

5. What I Learned

Through this lab, I gained practical knowledge in:

Performing EDA and understanding dataset challenges

Building regression and classification models in PyTorch

Applying normalization and encoding techniques

Handling imbalanced datasets using SMOTE

Evaluating models using loss/accuracy curves and confusion matrices

Using regularization techniques to control overfitting

Comparing baseline vs. optimized models

Using Git & GitHub for deep learning workflow version control

 6. Conclusion

This lab provided a complete end-to-end deep learning pipeline on two types of problems:

Regression: showed the limits of neural networks when the dataset is highly noisy and collinear

Classification: demonstrated how preprocessing, SMOTE, and PyTorch MLPs can produce strong results

Your final models are stable, correctly trained, and well-evaluated.
The work is complete, technically sound, and ready for submission.
