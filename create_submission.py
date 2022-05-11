import sys

import pandas as pd
from tqdm.auto import tqdm

method = sys.argv[1]


categories = {
    'D1': [],
    'D2': [],
    'D3': ['AE'],
    'D4': [],
    'D5': ['A', 'B', 'C', 'D', 'O'],
    'D6': ['A', 'B', 'C', 'D', 'O'],
    'D7': ['A', 'I'],
    'D8': ['A', 'B', 'D', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'],
    'D9': ['A', 'AB', 'BZ'],
    'D10': [],
}


dfs = {dataset: pd.read_csv(f"output/{method}/{dataset}.csv", dtype={c: 'category' for c in cats})
       for dataset, cats in categories.items()}


submission = pd.read_csv('input/sample_submission.csv')
submission['dataset'] = submission['cell_id'].apply(lambda x: x.split(',')[0])
submission['col'] = submission['cell_id'].apply(lambda x: x.split(',')[1])
submission['row'] = submission['cell_id'].apply(lambda x: int(x.split(',')[2]))


ans = []
for _, row in tqdm(submission.iterrows()):
    dataset = row['dataset']
    ans.append(dfs[row['dataset']].loc[row['row'], row['col']])


submission['value'] = ans
submission[['cell_id', 'value', 'type']].to_csv(f'submission_{method}.csv', index=False)
