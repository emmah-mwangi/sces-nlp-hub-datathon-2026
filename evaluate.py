import argparse,json
from pathlib import Path
import joblib,pandas as pd
from core import labels,fingerprint,classification_score,semantic_score
from validate import check
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/contribution.csv');p.add_argument('--track',choices=['intent','sentiment','semantic'],default='intent');p.add_argument('--practice',action='store_true');a=p.parse_args()
    df=check(a.data,a.practice,a.track);out=Path('artifacts')/a.track;pack=joblib.load(out/'model.joblib')
    if pack['practice']!=a.practice or pack['track']!=a.track or pack['data_sha256']!=fingerprint(df):raise ValueError('Dataset or run mode changed since training. Check and retrain deliberately.')
    test=df[df.split.eq('test')];ordered=labels(a.track)
    result={'track':a.track,'practice':a.practice,'test_size':len(test),'model':classification_score(pack['model'],test,ordered),'majority_baseline':classification_score(pack['dummy'],test,ordered)}
    pred=pack['model'].predict(test.text);test.assign(predicted=pred).to_csv(out/'test_predictions.csv',index=False)
    if a.track=='semantic':
        result['semantic']=semantic_score(pack['model'],pack['slot_memory'],test)
        unseen=test[test.slots_json.map(lambda raw:any(s['value'].casefold() not in pack['slot_memory'] for s in json.loads(raw)))]
        result['semantic']['unseen_value_test_count']=len(unseen)
        result['semantic']['unseen_value_scores']=semantic_score(pack['model'],pack['slot_memory'],unseen) if len(unseen) else None
    (out/'test_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nPRACTICE TEACHING DATA; NOT A BENCHMARK' if a.practice else '\nFINAL TEST RESULTS')
    print('Examples:',len(test))
    print(f"Accuracy: {result['model']['accuracy']:.3f} | Macro-F1: {result['model']['macro_f1']:.3f} | Majority macro-F1: {result['majority_baseline']['macro_f1']:.3f}")
    print('\nPer-label F1 (0 to 1; bars show proportion):')
    for label in ordered:
        row=result['model']['report'][label];bar='█'*round(row['f1-score']*20)
        print(f"{label:20s} {row['f1-score']:.3f} {bar:20s} n={int(row['support'])}")
    if a.track=='semantic':
        sem=result['semantic'];print(f"Slot F1: {sem['slot_f1']:.3f} | Full-frame exact match: {sem['frame_exact_match']:.3f}")
        print('Unseen-slot-value test examples:',sem['unseen_value_test_count'])
        if sem['unseen_value_scores'] is None:print('No unseen slot values tested; do not claim generalisation to them.')
    report=pd.DataFrame(result['model']['report']).T
    (out/'per_label_report.csv').write_text(report.to_csv(),encoding='utf-8')
    print(f'Files saved under {out}. Inspect test_predictions.csv. Do not tune against the test results.')

