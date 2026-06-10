import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Alice", "Bob", "Carol"],
    "Age": [25, 30, 35, 25, 30, 40],
})

# print(df, "\n")
# print(df[df.duplicated(subset="Name")], "\n")

df_clean = df.drop_duplicates(subset="Name")
# print(df_clean)

text = " Hello    world of  pandas!  "
# print(text.strip())

mask = df["Name"].str.lower() == df["Name"].str.lower()
print(mask)
print(type(mask))
print(df[mask])