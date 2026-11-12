import argparse,json
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from validate import check
from core import labels,fingerprint,classification_score,slot_memory,semantic_score
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/contribution.csv');p.add_argument('--track',choices=['intent','sentiment','semantic'],default='intent');p.add_argument('--practice',action='store_true');a=p.parse_args()
    df=check(a.data,a.practice,a.track);tr=df[df.split.eq('train')];dev=df[df.split.eq('dev')]
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4)),LogisticRegression(C=5.0,max_iter=1000,random_state=42));model.fit(tr.text,tr.label)
    dummy=DummyClassifier(strategy='most_frequent').fit(tr.text,tr.label)
    memory=slot_memory(tr) if a.track=='semantic' else {}
    pack={'model':model,'dummy':dummy,'language':df.language.iloc[0],'practice':a.practice,'track':a.track,'data_sha256':fingerprint(df),'slot_memory':memory}
    out=Path('artifacts')/a.track;out.mkdir(exist_ok=True,parents=True);joblib.dump(pack,out/'model.joblib')
    result={'track':a.track,'practice':a.practice,'model':classification_score(model,dev,labels(a.track)),'majority_baseline':classification_score(dummy,dev,labels(a.track))}
    if a.track=='semantic':result['semantic']=semantic_score(model,memory,dev)
    (out/'dev_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Saved {out}/model.joblib and dev_metrics.json. Dev macro-F1: {result['model']['macro_f1']:.3f}. Freeze choices before test evaluation.")

