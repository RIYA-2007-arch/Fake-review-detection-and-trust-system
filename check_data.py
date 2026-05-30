import pandas as pd

import pandas as pd

df = pd.read_csv("reviews.csv")
print(df['label'].value_counts())


print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape of dataset:")
print(df.shape)

print("\nLabel values:")
print(df['label'].value_counts())

