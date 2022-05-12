import pandas as pd

books = pd.read_csv('input/books.csv')
users = pd.read_csv('input/users.csv')
train_ratings = pd.read_csv('input/train_ratings.csv')
test_ratings = pd.read_csv('input/test_ratings.csv')


a_users = users['user_id'].unique()
mp_users = dict(zip(a_users, range(len(a_users))))

a_books = books['book_id'].unique()
mp_books = dict(zip(a_books, range(len(a_books))))

users['user_idx'] = users['user_id'].apply(lambda x: mp_users[x])
books['book_idx'] = books['book_id'].apply(lambda x: mp_books[x])

train_ratings['user_idx'] = train_ratings['user_id'].apply(lambda x: mp_users[x])
train_ratings['book_idx'] = train_ratings['book_id'].apply(lambda x: mp_books[x])
test_ratings['user_idx'] = test_ratings['user_id'].apply(lambda x: mp_users[x])
test_ratings['book_idx'] = test_ratings['book_id'].apply(lambda x: mp_books[x])


for c in ['city', 'province', 'country']:
    tmp = users.groupby(c).size().reset_index(name='sz')
    tmp[f'{c}_idx'] = tmp['sz'].rank(ascending=False, method='first').astype(int) - 1
    users = users.merge(tmp[[c, f'{c}_idx']], on=c)


for c in ['title', 'author', 'publisher']:
    tmp = books.groupby(c).size().reset_index(name='sz')
    tmp[f'{c}_idx'] = tmp['sz'].rank(ascending=False, method='first').astype(int) - 1
    books = books.merge(tmp[[c, f'{c}_idx']], on=c)


books.to_csv('input/transformed/books.csv', index=False)
users.to_csv('input/transformed/users.csv', index=False)
train_ratings.to_csv('input/transformed/train_ratings.csv', index=False)
test_ratings.to_csv('input/transformed/test_ratings.csv', index=False)
