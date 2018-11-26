from sklearn.preprocessing import Imputer
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

#mean of the column
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer = imputer.fit(df)
imputed_data = imputer.transform(df.values)
print(imputed_data)

