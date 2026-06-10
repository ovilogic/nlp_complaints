import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "charlie", "Alice", "Bob", "carol"],
    "Age": [25, 30, 35, 25, 30, 40],
})

# print(df, "\n")
# print(df[df.duplicated(subset="Name")], "\n")

df_clean = df.drop_duplicates(subset="Name")
# print(df_clean)

text = " Hello    world of  pandas!  "
# print(text.strip())

mask = df["Name"] == df["Name"].str.lower()
# print(mask)
# print(type(mask))
# print(df[mask])

s = pd.Series(["hello world", "one two three"])
# print(s.str.len())
# print(s.str.split().str.len())
# set = s.str.set()
# print(set)

# print(s)
# print(s.str.split().explode())

sentences = pd.Series(["Hello world", "Pandas is great.", "Data science is fun but challenging!"])
punctuation = sentences.str.contains(r"[^\w\s]", regex=True)
print(punctuation.sum())
