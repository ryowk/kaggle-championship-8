import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False, positive=True)

UID_NAME = "PassengerId"
TARGET_NAME = "Survived"

df_train = pd.read_csv('input/train.csv')[[UID_NAME, TARGET_NAME]]
df_test = pd.read_csv('input/test.csv')[[UID_NAME]]

output_dirs = list(Path('.').glob("*_model/output/*"))
print("output_dirs:", output_dirs)

for output_dir in output_dirs:
    train_paths = list(output_dir.glob('train_*.csv'))
    df = pd.concat([pd.read_csv(train_path) for train_path in train_paths])
    df = df.rename(columns={TARGET_NAME: str(output_dir)})
    df_train = df_train.merge(df, on=UID_NAME, validate='one_to_one')

    test_path = output_dir / 'test.csv'
    df = pd.read_csv(test_path)
    df = df.rename(columns={TARGET_NAME: str(output_dir)})
    df_test = df_test.merge(df, on=UID_NAME, validate='one_to_one')

X_train = df_train.drop([UID_NAME, TARGET_NAME], axis=1)
y_train = df_train[TARGET_NAME].values
model.fit(X_train, y_train)

model_names = df_train.drop([UID_NAME, TARGET_NAME], axis=1).columns
importance = {name: coef for name, coef in zip(model_names, model.coef_)}
with open("importance.json", 'w') as f:
    json.dump(importance, f, indent=4)

X_test = df_test.drop([UID_NAME], axis=1)
y_pred = model.predict(X_test)
df_test[TARGET_NAME] = y_pred

df_test[[UID_NAME, TARGET_NAME]].to_csv('submission.csv', index=False)
