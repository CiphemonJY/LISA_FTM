#!/usr/bin/env python3
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-v1', split='test')
print(f'wikitext-2-v1 test: {len(ds)} rows')
print('First 5 rows:')
for i, row in enumerate(ds):
    text = row["text"]
    print(f'  [{i}] len={len(text)} repr={repr(text[:80])}')
    if i >= 4: break
