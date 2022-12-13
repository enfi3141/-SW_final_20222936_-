##### 오픈소스SW_final_20222936_권유환
# Brain Tumor Classification_final project
-------------
### The reason I chose this model
##### [VotingClassifier]
> For random forest models, it is based on the decision tree algorithm. As a result, it is difficult to grasp the effect of each predictor, such as logistic regression, and it may lead to unstable predictions for new data. Conversely, logistic regression has limited expressiveness and performance problems. As such, there are pros and cons for each model.
> That is why I used ensemble learning to complement the shortcomings of single models by combining multiple models. Among them, I used a voting classifier in which classifiers with different algorithms are learned and combined based on the same dataset.
### Preprocessing used
---------------
##### [RobustScaler]
> I found that the model's performance was rather poor when I used minmax scaling and standard scaling. I analyzed that the cause of this is an outlier. So to solve this problem, I used robust scaling with median and IQR, and I was able to improve the performance of the model.
### Hyperparameter optimization
------------
> At this time, the hyperparameters were selected mainly to help improve model performance. In addition, for the n_jobs parameter,  fixed it to -1 to use the maximum number of CPU cores.
1. LogisticRegression
    - solver
      - Since the training dataset has four classes and has a lot of data, I compared saga, an algorithm suitable for large data, and models newton-cg and lbfgs, which are suitable for multi-class classification models.
    - penalty
      - Since it is a hyperparameter that sets the constraints, it was compared using l2, elasticnet, and none. However, l1 was not used to collide with newton-cg and lbfgs.
    - max_iter
      - This does not affect the performance results after convergence with the number of tasks to be used for calculation. Therefore, it was only checked when it converged and showed maximum performance without putting in an excessively large number.
    - C
      - It means a cost function, and the smaller the number, the greater the limit, so the overall number was compared using logspace.
- using code
```python
from sklearn.model_selection import GridSearchCV
logModel = sklearn.linear_model.LogisticRegression()
param_grid = [    
    {'penalty' : ['l2','elasticnet','none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['newton-cg','lbfgs','saga'],
    'max_iter' : [100, 1000,2000, 5000]
    }
]
model = GridSearchCV(logModel,param_grid=param_grid,n_jobs=-1)
best_model = model.fit(X,y)
best_model.best_estimator_
```
2. RandomForestClassifier
    - n_estimators
      - Since this is a hyperparameter related to the number of trees to be generated, it is not good to be too large, so I applied it to the model up to 200.
    - random_state
      - The hyperparameters determining the random number were fixed at 100.
- using code
```python
n_estimators = [50,100,150,200]
for n in n_estimators:
  rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=n, random_state=100,n_jobs = -1)
  rfc = rfc.fit(X_train_scaled, y_train)
  y_pred = rfc.predict(X_test_scaled)
  print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```
3. KNeighborsClassifier
    - n_neighbors
      - The number of neighbors used to spin the model was determined from 1 to 10.
- using code
```python
n_neighbors = [1,2,3,4,5,6,7,8,9,10]
for n in n_neighbors:
  knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
  knn = knn.fit(X_train_scaled, y_train)
  y_pred = knn.predict(X_test_scaled)
  print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```
### Check model performance
---------------
- using code
```python
print(sklearn.metrics.classification_report(y_test,y_pred))
```
![image](https://user-images.githubusercontent.com/115198568/207351995-9d64f30a-1b7c-41de-a25f-62b70a00cd49.png)
> By comparing the precision and recall rate, it was possible to check whether there was a large difference between the actual value and the predicted value. In addition, the weighted harmonized average of precision and recall could be seen through f1-score. Through this, it was found that this model works to produce good performance.
