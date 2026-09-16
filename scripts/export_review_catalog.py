"""Enumerate noun forms and construction examples for human reviewers."""
import argparse
import csv
from pathlib import Path
from flystudy.data import NOUNS,COLORS,SIZES,LANGUAGES,SPLITS,noun_phrase,particle,load_rows
from flystudy.protocol import file_hash,write_json

p=argparse.ArgumentParser(); p.add_argument('--data',required=True); a=p.parse_args()
root=Path(a.data)
with (root/'noun-forms.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['language','noun','color','size','case','count_index','surface'])
    for lang in LANGUAGES:
        for n in range(len(NOUNS)):
            for c in range(len(COLORS)):
                for s in range(len(SIZES)):
                    for case in ('nom','acc','dat'):
                        value=noun_phrase((n,c,s),lang,case)
                        if lang=='ko' and case in ('nom','acc'):
                            value=particle(value,('이','가') if case=='nom' else ('을','를'))
                        w.writerow([lang,n,c,s,case,'',value])
                    for count in range(7):
                        w.writerow([lang,n,c,s,'plural',count,noun_phrase((n,c,s),lang,count=count)])
with (root/'construction-examples.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['language','split','task','verb','neg','label','sentence_a','sentence_b'])
    used=set()
    for split in SPLITS:
        for r in load_rows(root,split):
            key=(r['language'],split,r['task'],r['scene']['verb'],r['scene']['neg'],r['label'])
            if key not in used:
                used.add(key); w.writerow([*key,r['sentence_a'],r['sentence_b']])
write_json(root/'review-catalog.json',{'status':'awaiting_human_review',
    'files':{name:file_hash(root/name) for name in ('noun-forms.csv','construction-examples.csv','template-inventory.json')},
    'note':'All noun surface forms plus construction examples; also inspect the renderer and verb table for full template review.'})
