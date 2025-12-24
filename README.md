Pandas fundamental codes

🔹 1. Import & Load Data
import pandas as pd

pd.read_csv("file.csv")
pd.read_excel("file.xlsx")

🔹 2. Create Data
pd.Series([1, 2, 3])
pd.DataFrame({"A": [1, 2], "B": [3, 4]})

🔹 3. Inspect Data (MOST USED)
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.columns

🔹 4. Select Data
df["col"]
df[["col1", "col2"]]

df.loc[0]
df.iloc[0]

df[df["Marks"] > 80]

🔹 5. Change / Add Columns
df["New"] = df["Old"] * 2

🔹 6. Datatype Handling (IMPORTANT)
.astype()
df["Marks"] = df["Marks"].astype(int)
df["Gender"] = df["Gender"].astype("category")

🔹 7. Categorical Encoding
.cat.codes
df["Gender"] = df["Gender"].astype("category")
df["Gender"] = df["Gender"].cat.codes


⚠️ -1 → missing value

🔹 8. Missing Values
df.isnull()
df.isnull().sum()

df.fillna(0)
df.dropna()

🔹 9. Sorting
df.sort_values("Marks")
df.sort_values("Marks", ascending=False)

🔹 10. Aggregations
df["Marks"].mean()
df["Marks"].max()
df["Marks"].min()
df["Marks"].sum()
df["Marks"].count()

🔹 11. GroupBy (POWER FEATURE)
df.groupby("Category")["Marks"].mean()
df.groupby("Category").count()

🔹 12. Apply / Vectorization
df["Marks"] + 5
df["Marks"].apply(lambda x: x + 5)

🔹 13. Rename & Drop
df.rename(columns={"Marks": "Score"})
df.drop("Column", axis=1)
df.drop(0)

🔹 14. Merge / Join
pd.merge(df1, df2, on="ID")
pd.merge(df1, df2, how="left")

🔹 15. One-Hot Encoding (Alternative to .cat.codes)
pd.get_dummies(df["City"])

🔹 16. Export Data
df.to_csv("out.csv", index=False)
df.to_excel("out.xlsx", index=False)
