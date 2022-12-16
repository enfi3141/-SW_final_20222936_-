##### 오픈소스SW_final_20222936_권유환
# Brain Tumor Classification_final project
-------------
### The reason I chose this model
##### [VotingClassifier]
> Knn uses a lot of memory and takes a long time, unlike logistic regression because it compares all the existing data. Conversely, logistic regression has limited expressiveness and performance problems. As such, there are pros and cons for each model.
> That is why I used ensemble learning to complement the shortcomings of single models by combining multiple models. Among them, I used a voting classifier in which classifiers with different algorithms are learned and combined based on the same dataset.
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
2. KNeighborsClassifier
    - n_neighbors
      - The number of neighbors used to spin the model was determined from 1 to 20.
    - metric
      - Since it is a measure of distance calculation, all cases were compared.
    - weights
      - Since it is a parameter that gives different weights depending on the distance of adjacent samples, it is thought to play an important role in performance, so it was included in the comparison.
    - leaf_size
      - If it is too small, the performance decreases, and if it is too large, the performance decreases again due to noise, so I put a certain range value.
    - p
      - Since it is a parameter, I put 1 and 2.
- using code
```python
from sklearn.model_selection import GridSearchCV
grid_params = {
    'n_neighbors' : list(range(1,20)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski'],
    'leaf_size' : list(range(1,20)),
    'p' : [1, 2]
}
gs = GridSearchCV(knn, grid_params, cv=10)
gs.fit(X_train, y_train)
print("Best Parameters : ", gs.best_params_)
```
### Check model performance
---------------
- using code
```python
print(sklearn.metrics.classification_report(y_test,y_pred))
```
![image](https://user-images.githubusercontent.com/115198568/208087936-40a888ed-27b5-4258-b78a-6854d1f066f3.png)
> By comparing the precision and recall rate, it was possible to check whether there was a large difference between the actual value and the predicted value. In addition, the weighted harmonized average of precision and recall could be seen through f1-score. Through this, it was found that this model works to produce good performance.
