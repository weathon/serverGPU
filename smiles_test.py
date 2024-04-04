import pandas as pd
test_cids = [8100, 7298, 40555, 13007, 61929, 93058, 220010, 14020, 97502]
print(len(test_cids))
molecules_rows = pd.read_csv("summary.csv")
test = molecules_rows[molecules_rows[" cid"].isin(test_cids)]
val = molecules_rows[~(molecules_rows[" cid"].isin(test_cids))]
print(set(test["canonicalsmiles"]).intersection(set(val["canonicalsmiles"])))
