import spacy

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")
# doc = nlp("This is a sentence.")
# print(doc.text)  # Output: This is a sentence. 

# for token in doc:
#     print(token.text, token.pos_, token.dep_, sep='---')
#     print(token.lemma_, token.tag_, token.shape_, token.is_alpha,
#            token.is_stop, token.is_punct, end='\n\n')


# print(nlp.pipe_names)

# doc = nlp("Apple is a US company with a valuation second only to Nvidia.")
# for ent in doc.ents:
#     print(ent.text, ent.label_)

# for i in ["GPE", "ORG", "ORDINAL"]:
#     print(spacy.explain(i))

# for token in doc:
#     print(token.text, token.pos_, token.tag_)

# root = [token for token in doc if token.dep_ == "ROOT"][0]
# # print(root)
# # for child in root.children:
# #     print(child.text, child.dep_)

# for token in root.subtree:
#     print(token.text)

# doc = nlp("This is the first sentence. This is the second. And now the third.")
# # for sent in doc.sents:
# #     print(sent.text)

# span = doc[0:13]
# print(span)
# counter = 1
# for i in span:
#     print(str(counter) + " --- ", i)
#     counter += 1

# for ent in doc.ents:
#     print(ent.text, ent.start, ent.end, ent.label_)

# doc1 = nlp("I like cats")
# doc2 = nlp("I love dogs")

# print(doc1.similarity(doc2))

wd1 = nlp("dog")
wd2 = nlp("canine")
# print(wd1.similarity(wd2))

token = wd1[0]
# print(token.vector)

