"""T29: choose model settings using DEV data only. Never uses the test split."""
import itertools
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from validate import check
from core import labels, classification_score

TRACK = 'intent'
DEFAULT = ('char', (2, 5), 1.0)   # what train.py uses now
MIN_GAIN = 0.02                   # only change settings for a clear dev gain

df = check('data/contribution.csv', False, TRACK)
tr = df[df.split.eq('train')]
dev = df[df.split.eq('dev')]
print(f'train={len(tr)} dev={len(dev)} (test rows are not used)')

grid = [DEFAULT] + [
    g for g in itertools.product(['char', 'char_wb'],
                                 [(1, 4), (2, 4), (2, 5), (3, 6)],
                                 [0.5, 1.0, 5.0, 10.0, 50.0])
    if g != DEFAULT
]

rows = []
for analyzer, ngram, C in grid:
    model = make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram),
        LogisticRegression(C=C, max_iter=1000, random_state=42))
    model.fit(tr.text, tr.label)
    s = classification_score(model, dev, labels(TRACK))
    rows.append({'analyzer': analyzer, 'ngram_range': str(ngram), 'C': C,
                 'dev_macro_f1': round(s['macro_f1'], 4),
                 'dev_accuracy': round(s['accuracy'], 4)})

res = pd.DataFrame(rows)
res.to_csv('docs/dev_tuning_results.csv', index=False)

ranked = res.sort_values(['dev_macro_f1', 'dev_accuracy'], ascending=False, kind='stable')
print('\nTop 8 settings on dev:')
print(ranked.head(8).to_string(index=False))

default_f1 = res.iloc[0].dev_macro_f1
best = ranked.iloc[0]
print(f'\nDefault (char, (2, 5), C=1.0): dev macro-F1 = {default_f1}')
print(f'Best: {best.analyzer}, {best.ngram_range}, C={best.C}: dev macro-F1 = {best.dev_macro_f1}')
if best.dev_macro_f1 - default_f1 >= MIN_GAIN:
    print('DECISION: CHANGE settings to the best row above.')
else:
    print(f'DECISION: KEEP DEFAULT (gain is below {MIN_GAIN}, within dev noise).')