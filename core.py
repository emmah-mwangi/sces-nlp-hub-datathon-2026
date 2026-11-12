"""Shared small-data baselines; no paid APIs or automatic translation."""
import json, re, hashlib
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

INTENTS=['weather_query','datetime_query','alarm_set','calendar_set','calendar_query','lists_createoradd','lists_query','play_music','news_query','qa_definition']
SLOT_TYPES={'location','date','time','item','term'}
FRIENDLY={'weather_query':'Weather enquiry','datetime_query':'Date or time enquiry','alarm_set':'Set alarm','calendar_set':'Create calendar event','calendar_query':'Read calendar','lists_createoradd':'Create or add to list','lists_query':'Read list','play_music':'Play music','news_query':'News enquiry','qa_definition':'Word or concept meaning','positive':'Positive feedback','negative':'Negative feedback','neutral':'Neutral statement'}
def labels(track):
    return ['positive','negative','neutral'] if track=='sentiment' else INTENTS
def fingerprint(df):
    return hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()
def slot_memory(train):
    counts=Counter()
    for text, raw in zip(train.text,train.slots_json):
        for s in json.loads(raw): counts[(s['value'].casefold(),s['type'])]+=1
    # One deterministic type per known surface form; use training data only.
    values={}
    for (value, kind), count in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        values.setdefault(value,kind)
    return values
def extract(text,memory):
    candidates=[]
    for value, kind in memory.items():
        for match in re.finditer(r'(?<!\w)'+re.escape(value)+r'(?!\w)',text,re.IGNORECASE):
            candidates.append({'type':kind,'start':match.start(),'end':match.end(),'value':match.group()})
    accepted=[]
    for s in sorted(candidates,key=lambda s:(-(s['end']-s['start']),s['start'],s['type'])):
        if not any(s['start']<a['end'] and a['start']<s['end'] for a in accepted):accepted.append(s)
    return sorted(accepted,key=lambda s:s['start'])
def slot_keys(slots):
    return {(s['type'],s['start'],s['end']) for s in slots}
def classification_score(model,frame,ordered):
    pred=model.predict(frame.text)
    return {'accuracy':accuracy_score(frame.label,pred),'macro_f1':f1_score(frame.label,pred,labels=ordered,average='macro',zero_division=0),
    'report':classification_report(frame.label,pred,labels=ordered,output_dict=True,zero_division=0),
    'confusion_matrix':confusion_matrix(frame.label,pred,labels=ordered).tolist(),'label_order':ordered}
def semantic_score(model,memory,frame):
    intents=model.predict(frame.text);tp=fp=fn=exact=0;predictions=[]
    for intent, (_,row) in zip(intents,frame.iterrows()):
        got=extract(row.text,memory);gold=json.loads(row.slots_json)
        a=slot_keys(got);b=slot_keys(gold);tp+=len(a&b);fp+=len(a-b);fn+=len(b-a)
        exact+=int(intent==row.label and a==b)
        predictions.append({'record_id':row.record_id,'intent':intent,'slots':got})
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {'slot_precision':precision,'slot_recall':recall,'slot_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'frame_exact_match':exact/len(frame),'predictions':predictions,
            'definition':'Exact slot type and character span; frame requires correct intent and complete slot set.'}

