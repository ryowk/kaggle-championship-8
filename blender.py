import pandas as pd
import numpy as np

paths = [
    'wktk_0_model/output/sub0/prob.csv',
    'wktk_0_model/output/sub1/prob.csv',
    'wktk_0_model/output/sub3/prob.csv',
]

dfs = [pd.read_csv(path) for path in paths]
for i, df in enumerate(dfs):
    dfs[i] = df.sort_values('image_id').reset_index(drop=True)
    dfs[i] = dfs[i].drop('image_id', axis=1).values

vals = [a for a in dfs]

p = np.mean(vals, axis=0)
df_pred = pd.read_csv(paths[0])[['image_id']].sort_values('image_id').reset_index(drop=True)
df_pred['class_6'] = np.argmax(p, axis=1)
df_pred.to_csv('submission.csv', index=False)

df_sample = pd.read_csv('input/sample_submission.csv')
assert set(df_pred['image_id'].values) == set(df_sample['image_id'].values)
