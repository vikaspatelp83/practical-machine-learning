# KNN - k nearest neighbor classifier code in Python

### Import the libraries


```python
from sklearn import datasets # dataset
from sklearn.neighbors import KNeighborsClassifier #knn classifier 
```

#### load dataset


```python
iris = datasets.load_iris()
```

#### Print features and targets


```python
features = iris.data
labels = iris.target

print("features =", features[0])
print("labels =",labels[0])
```

    features = [5.1 3.5 1.4 0.2]
    labels = 0
    

##### description of above result  

Labels or 
Classes :
* 0 -> iris setosa
* 1 -> iris versicolor
* 2 -> iris verginica

Features : by their occurence
* sepal length
* sepal width
* petal length
* petal width

#### Classifier train


```python
clf = KNeighborsClassifier()
clf.fit(features,labels)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



##### Predict new data 

> clf.predict ( [ [ sepal_length, sepal_width, petal_length, petal_width ] ] )

Classes :
* [0] -> iris setosa
* [1] -> iris versicolor
* [2] -> iris verginica

> classes = {0:"Iris Setosa", 1:"Iris Versicolor", 2:"Iris Verginica"}


```python
classes = {0:"Iris Setosa", 1:"Iris Versicolor", 2:"Iris Verginica"}
```


```python
pred = clf.predict ([[1,1,1,1]])
print(pred)
```

    [0]
    

##### print name of the class


```python
print(classes[0])
```

    Iris Setosa
    

##### print names of the classes 
### defining a function


```python
def print_label(found_labels,xdata):
    i = 0 
    print("Class\t\t->\t\tData\n")
    for label_id in found_labels:
        print(classes[label_id],"\t->\t",xdata[i])
        i += 1
```

### Let's classify a list of new data

#### Data is here 
> data is a two dimensional array

```
data = [
            [ sepal_length, sepal_width, petal_length, petal_width ],
            [ sepal_length, sepal_width, petal_length, petal_width ],
            [ sepal_length, sepal_width, petal_length, petal_width ],
            .
            .
            .
            [ sepal_length, sepal_width, petal_length, petal_width ]
        ]
```



```python
data = [
    [2,4,2,4],
    [3,5,1,7],
    [6,2,6,3],
    [5.3,7.5,25,9.7],
    [5.3,7.5,25,19.7],
    [2,4,2,4],
    [3,5,1,7],
    [6,2,6,3],
    [5.3,7.5,25,9.7],
    [5.3,7.5,25,19.7]
    
]
```

##### Predict


```python
# predict
prediction = clf.predict(data)
```


```python
print(prediction)
found_labels = list(prediction)
```

    [0 0 2 2 2 0 0 2 2 2]
    

##### print the classes according to the data


```python
print_label(found_labels,data)
```

    Class		->		Data
    
    Iris Setosa 	->	 [2, 4, 2, 4]
    Iris Setosa 	->	 [3, 5, 1, 7]
    Iris Verginica 	->	 [6, 2, 6, 3]
    Iris Verginica 	->	 [5.3, 7.5, 25, 9.7]
    Iris Verginica 	->	 [5.3, 7.5, 25, 19.7]
    Iris Setosa 	->	 [2, 4, 2, 4]
    Iris Setosa 	->	 [3, 5, 1, 7]
    Iris Verginica 	->	 [6, 2, 6, 3]
    Iris Verginica 	->	 [5.3, 7.5, 25, 9.7]
    Iris Verginica 	->	 [5.3, 7.5, 25, 19.7]
    

#### create a large dataset


```python
new_data =[]
import random
for i in range(400):
    new_data.append(random.randrange(1, 20, 1))
```

##### Reshape the data into ( 4, 1 ) size


```python
import numpy as np
mdata = np.array(new_data)
ndata = mdata.reshape(100,4)
```

##### predict the result


```python
prediction = clf.predict(ndata)
```

##### print names of classes


```python
#print_label(list(prediction),list(ndata))

```
