# machine-learning-in-python
Machine Learning in Python Workspace

# DataPreProcessingLibrary.py Info

DataPreProcessingLibrary is basic data pre-processing library in python. We utilize this small library to help data pre processing methods in machine learning samples. 

It is a class which utilizes 'pandas' library to provide to load raw data from CSV files. 
```python
class RawDataLoader:
    
    raw_data = None
    
    def load(self, path):
        self.raw_data = pd.read_csv(path)
        return self.raw_data
    
    def getData(self):
        return self.raw_data
```

It is a class which utilizes 'scikit-learn' library to complete missing values in data set.
```python
class MissingValueImputer:
    
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    
    def fill_missing_values(self, raw_data, startColumnIndex, endColumnIndex):
        ...
```

It is a class which utilizes 'scikit-learn' library to convert categorical data into numeric values.
```python
class CategoricalDataProcessor:
    
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    
    def makeLabelEncode(self, data):
        ...
    
    def makeOneShotEncoding(self, data):
        ...
```

It is a class which utilizes 'pandas' library to concat data parts which are formatted as DataFrame.
```python
class DataImporter:
    
    def createDataFrame(self, data, rowSize, columns):
        ...
    
    def concat(self, dataParts, axisWay):
        ...
    
    def createColumnNames(self, prefix, size):
        ...
```

It is a class which utilizes 'scikit-learn' library to split data into small chunks.
```python
class DataSplitter:
    
    def split(self, data, target_labels, testSize):
        
        return ms.train_test_split(data, target_labels, test_size = testSize, random_state = 0)
```

It is a class which utilizes 'scikit-learn' library provides to scale data fields. Also, you can normalize data fields.
```python
class DataScaler:
    
    def scale(self, data):
        standardScaler = preprocessing.StandardScaler()
        return standardScaler.fit_transform(data)
    
    def normalize(self, data):
        minMaxScaler = preprocessing.MinMaxScaler()
        return minMaxScaler.fit_transform(data)
```

Sometimes, we need to reduce the size of data in terms of columns. We dont want to keep unnecessary fields in data set. So in this basic algorithm, we applied 'BackwardElimination' technic to reduce the data columns. 

'performBackwardElimination' is a base method to appy the backward elimination.
'significanceLevel' is error significance level.
'onestep' might get True or False values. If value is True, backward elimination is applied sep by step. It removes the field which has highest P-Value in data set. Then, it iterates same steps.

```python
class DimensionReductionTool:
    
    model = None
    significanceLevel = 0.05
    
    def __init__(self, significanceLevel):
        self.significanceLevel = significanceLevel
        
    def performBackwardElimination(self, targetLabels, fulldata, onestep=False):
        ...
```

# Regression Model Results

```console
Linear Regression >> R2 Value: 0.8003564455001666
Polynomial Regression >> R2 Value: 0.5641489621110274
SVR >> R2 Value: 0.9077537607951016
Decision Tree >> R2 Value: 0.9791666666666667
Random Forest >> R2 Value: 0.8476025208333335
```
