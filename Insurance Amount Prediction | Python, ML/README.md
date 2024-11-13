# Analysis Summary:
In our two-phase analysis of predicting insurance amounts, we explored multiple models to identify the best approach for different age groups and closely examined the accuracy of predictions, particularly focusing on age-based discrepancies.

#### Phase 1 Summary: 
After training several models on the combined dataset, XGB Regression emerged as the top-performing model, achieving an R² score of 98.2%. However, despite this high R² score, the model had a relatively high mean squared error (MSE), and a deeper analysis revealed that approximately 30% of consumers were either overcharged or undercharged, with discrepancies ranging from 10% to 90%. These inconsistencies were particularly pronounced in the 18 to 25 age group, indicating that the model struggled with accurate predictions for younger individuals. This observation prompted a more granular analysis, where the dataset was split into two segments: individuals below 26 and those above 25.

#### Phase 2 Summary: 
Upon segmenting the data, we observed clear differences in model performance between the two age groups. For individuals below 26, Lasso Regression yielded the best performance among the tested models, but with an R² score of only 60%, indicating a relatively low accuracy and poor predictive capability for this age group. In contrast, for individuals aged above 25, the XGB Regressor—enhanced with hyperparameter tuning—achieved a very high R² score of 99.8%, showcasing strong predictive accuracy and a solid model fit.

##### KDE Plot and Difference Percentage Analysis: 
The KDE plot analysis highlighted that predictions for individuals below 26 deviated significantly from actual values, further confirming the model's subpar performance in this segment. However, for individuals aged above 25, the KDE plot showed close alignment between predicted and actual values, indicating a well-performing model for this group. Further analysis of the percentage differences revealed that over 73% of individuals below 26 were either overcharged or undercharged, with discrepancies ranging from 10% to 90%. For the 25+ age group, however, the model maintained consistency, with no individuals exceeding the established 10% error margin.


#### Recommendations for Improving Predictions for Individuals Below 26:

##### Feature Engineering:
Create new features that may better capture the unique factors affecting younger individuals’ insurance costs, such as lifestyle indicators, risk factors, and employment status. Adding non-linear transformations or interactions between features might also help the model better capture the underlying patterns for this age group.

##### Age-Specific Model Tuning:
Develop a separate model specifically tuned for individuals below 26, rather than using a single model across all ages. This approach allows for more tailored hyperparameter tuning and model selection to optimize performance for younger individuals.

##### Incorporating External Data:
Supplement the existing dataset with external data sources that could provide insights into younger consumers’ behaviors and risks. Data on spending habits, healthcare visits, or even regional information could help refine predictions for this group.

##### Ensemble Methods:
Implement an ensemble model that combines the strengths of different algorithms, such as blending Lasso Regression, Decision Trees, and XGB Regressor for age-specific predictions. Ensemble methods could help balance the predictive strengths across different types of data distributions.

##### Targeted Hyperparameter Tuning:
Perform an extensive hyperparameter search for models specific to the below-26 segment, focusing on parameters that control model complexity (e.g., regularization for Lasso Regression or maximum depth for tree-based models).

##### Handling Outliers and High Variance in Predictions:
Identify and address any outliers in the below-26 dataset that might be skewing predictions. Techniques like quantile regression or robust regression could be used to minimize the influence of extreme values on the model.

#### Final Insights: 
Our analysis indicates that a one-size-fits-all approach may not be ideal for predicting insurance costs across different age groups. While the XGB Regressor performs well for individuals above 25, separate modeling and targeted tuning are necessary for younger consumers to improve prediction accuracy. By implementing age-specific models and incorporating additional features or external data, we can enhance the model’s ability to make fair and accurate predictions across all age groups, ensuring more precise and equitable insurance pricing.
