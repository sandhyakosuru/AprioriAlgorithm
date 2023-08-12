import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('C:/Users/91901/Downloads/dataset.csv', header=None)
print(data.info())
print(data)
data = list(data[0].apply(lambda x: x.split(',')))
print(data)

te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data, columns=te.columns_)
print(df)

df1 = apriori(df, min_support=0.4, use_colnames=True)
print(df1)
print(df1.sort_values(by="support", ascending=False))

df_ar = association_rules(df1, metric="confidence", min_threshold=0.5)
print(df_ar)