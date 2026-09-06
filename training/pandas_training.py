import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "charlie", "NATO", "BOB", "carol", "jamie Dyer"],
    "Age": [25, 30, 35, 25, 30, 40, 43],
})

print(df.max())

# print(df.columns.get_loc("Name"))
# for i in df.columns:
#     print(i, df.columns.get_loc(i))
# print(df, "\n")
# print(df[df.duplicated(subset="Name")], "\n")

# df_clean = df.drop_duplicates(subset="Name")
# # print(df_clean)

# text = " Hello    world of  pandas!  "
# # print(text.strip())

# mask = df["Name"] == df["Name"].str.lower()
# print(mask)
# print(type(mask))
# print(df[mask])

# upper_mask = (df["Name"] == df["Name"].str.upper()).sum()
# print(upper_mask)

# s = pd.Series(["hello world", "one two three"])
# print(s.str.len())
# print(s.str.split().str.len())
# set = s.str.set()
# print(set)

# print(s)
# print(s.str.split().explode())

# sentences = pd.Series(["Hello world  ", "Pandas is   great.", "   Data science is   fun but   challenging!"])
# punctuation = sentences.str.contains(r"[^\w\s]", regex=True) # remember ^ inside a character class, [], is a negation operator, not the start of a string.
# print(punctuation)
# # print(punctuation.sum())
# stripped = sentences.str.strip()
# print(stripped)

# df["Location"] = ["New York", "Los Angeles", "Chicago", "New York", "Los Angeles", "Chicago"]
# print(df[["Name", ]])
