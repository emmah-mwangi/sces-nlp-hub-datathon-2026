"""Structural checks cannot replace fluent review, permission evidence or semantic duplicate review."""
import argparse,json,re,unicodedata
from collections import Counter
import pandas as pd
from core import labels,SLOT_TYPES
FIELDS=['record_id','language','text','label','source_dataset','source_version','source_record_id','source_url','source_method','licence_or_permission','contributor_code','reviewer_code','review_status','group_id','split']
def check(path,practice=False,track='intent'):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if 'label' not in df and 'intent' in df:df=df.rename(columns={'intent':'label'})
    required=FIELDS+(['slots_json'] if track=='semantic' else [])
    missing=set(required)-set(df.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if df.empty or df[required].map(lambda x:not x.strip()).any().any():raise ValueError('Required fields cannot be empty; explain justified not_applicable values.')
    if df.record_id.duplicated().any():raise ValueError('Duplicate record IDs.')
    key=df.text.map(lambda x:re.sub(r'\s+',' ',unicodedata.normalize('NFC',x).strip().casefold()))
    if key.duplicated().any():raise ValueError('Duplicate text after basic normalisation.')
    if set(df.label)-set(labels(track)) or set(df.split)-{'train','dev','test'}:raise ValueError('Unknown label or split.')
    if df.language.nunique()!=1:raise ValueError('Use one primary language per contribution file.')
    if df.groupby('group_id').split.nunique().gt(1).any():raise ValueError('Related examples cross splits; keep each group_id together.')
    if df.text.str.contains(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?:\+254|0)[17]\d{8}',regex=True).any():raise ValueError('Possible private email or phone in text; inspect.')
    for split in ['train','dev','test']:
        if set(df.loc[df.split.eq(split),'label'])!=set(labels(track)):raise ValueError('Every split must contain every label: '+split)
    if track=='semantic':
        slot_rows=0
        for _,row in df.iterrows():
            try:slots=json.loads(row.slots_json)
            except Exception as e:raise ValueError('Invalid slots_json: '+row.record_id) from e
            if not isinstance(slots,list):raise ValueError('slots_json must contain a JSON list.')
            slot_rows+=bool(slots);used=[]
            for s in slots:
                if not isinstance(s,dict) or set(s)!={'type','value','start','end'}:raise ValueError('Slot needs type,value,start,end.')
                if s['type'] not in SLOT_TYPES or type(s['start']) is not int or type(s['end']) is not int:raise ValueError('Invalid slot type or index.')
                if not 0<=s['start']<s['end']<=len(row.text) or row.text[s['start']:s['end']]!=s['value']:raise ValueError('Slot span/value differs from text: '+row.record_id)
                if any(s['start']<b and a<s['end'] for a,b in used):raise ValueError('Overlapping/duplicate slots are outside this starter schema.')
                used.append((s['start'],s['end']))
        if not practice and slot_rows<100:raise ValueError('Semantic track needs at least 100 records with a non-empty slot annotation.')
    if not practice:
        if set(df.source_method)-{'original_human','human_localised','newly_annotated','corrected_with_evidence'}:raise ValueError('Invalid contribution method; unchanged downloads and practice rows do not count.')
        if set(df.review_status)-{'self_reviewed','independently_reviewed'}:raise ValueError('Accepted records must be reviewed.')
        minima={'train':60,'dev':20,'test':20} if track=='sentiment' else {'train':15,'dev':5,'test':5}
        target=300 if track=='sentiment' else 250
        if len(df)<target:raise ValueError(f'This track requires at least {target} accepted contribution records.')
        for split,minimum in minima.items():
            counts=Counter(df.loc[df.split.eq(split),'label'])
            if any(counts[x]<minimum for x in labels(track)):raise ValueError(f'{split}: need at least {minimum} per label.')
        test=df[df.split.eq('test')]
        if not test.review_status.eq('independently_reviewed').all():raise ValueError('Every test item needs independent review.')
        reviewed=df.review_status.eq('independently_reviewed')
        if (df.loc[reviewed,'contributor_code']==df.loc[reviewed,'reviewer_code']).any():raise ValueError('Reviewer and author must differ.')
    return df
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/contribution.csv');p.add_argument('--track',choices=['intent','sentiment','semantic'],default='intent');p.add_argument('--practice',action='store_true');a=p.parse_args()
    df=check(a.data,a.practice,a.track);print('PASS:',len(df),'records; language quality, permission and semantic duplicate review remain human responsibilities.')

