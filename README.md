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

It is a class which utilizes 'scikit-learn' librar to complete missing values in data set.
```python
class MissingValueImputer:
    
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    
    def fill_missing_values(self, raw_data, startColumnIndex, endColumnIndex):
        ...
```
