import argparse
import json
import platform
import statistics
from pathlib import Path
from .domain import LABELS,PROMPT_VERSION,fingerprint

def metrics(rows):
    if not rows: raise ValueError("Empty evaluation")
    matrix=[[0]*8 for _ in LABELS]
    for r in rows:
        if r['expected'] not in LABELS: raise ValueError('Unknown expected label')
        matrix[LABELS.index(r['expected'])][LABELS.index(r['label']) if r['label'] in LABELS else 7]+=1
    report={}
    for i,label in enumerate(LABELS):
        tp=matrix[i][i]; support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0; recall=tp/support if support else 0
        report[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'support':support}
    times=sorted(r['latency_ms'] for r in rows if r.get('latency_ms') is not None)
    return {'accuracy':sum(matrix[i][i] for i in range(7))/len(rows),
            'macro_f1':sum(r['f1'] for r in report.values())/7,'classification_report':report,
            'confusion_matrix':{'rows':LABELS,'columns':[*LABELS,'INVALID'],'values':matrix},
            'invalid_count':sum(row[7] for row in matrix),'count':len(rows),
            'latency_median_ms':statistics.median(times) if times else None}

def evaluate(classifier,rows,model_id,strict=False):
    predictions=[]
    for row in rows:
        try:
            prediction=classifier.predict(row['text'])
            error=None
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Inference failed for case {row['id']}; evaluation is incomplete") from exc
            prediction={'label':None,'latency_ms':None}
            error=type(exc).__name__
        predictions.append({'id':row['id'],'expected':row['label'],**prediction,'error':error})
    return {'schema_version':'1','task_type':'support_ticket_classification','model_id':model_id,
            'dataset_fingerprint':fingerprint(rows),'prompt_version':PROMPT_VERSION,
            'python':platform.python_version(),'metrics':metrics(predictions),'predictions':predictions}

def compare(a,b):
    for report in (a,b):
        if not report['predictions'] or any(r.get('error') for r in report['predictions']):
            raise ValueError('Cannot compare incomplete evaluations containing inference exceptions')
    for key in ('dataset_fingerprint','prompt_version'):
        if a[key]!=b[key]: raise ValueError(f'Unmatched comparison: {key}')
    if [(r['id'],r['expected']) for r in a['predictions']] != [(r['id'],r['expected']) for r in b['predictions']]:
        raise ValueError('Unmatched cases')
    return {'task_type':'support_ticket_classification','baseline':a['model_id'],'candidate':b['model_id'],
            'dataset_fingerprint':a['dataset_fingerprint'],
            'accuracy_delta':b['metrics']['accuracy']-a['metrics']['accuracy'],
            'macro_f1_delta':b['metrics']['macro_f1']-a['metrics']['macro_f1'],
            'note':'Latency includes tokenization and generation, excludes model loading. No cost claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run'); run.add_argument('--model',required=True); run.add_argument('--data',default='data/prepared/validation.json'); run.add_argument('--output',required=True)
    pair=sub.add_parser('compare'); pair.add_argument('baseline'); pair.add_argument('candidate'); pair.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise ValueError('Output already exists; use a new run filename')
    if a.command=='run':
        from .inference import HFClassifier
        result=evaluate(HFClassifier(a.model),json.loads(Path(a.data).read_text(encoding='utf-8')),a.model,strict=True)
    else: result=compare(json.loads(Path(a.baseline).read_text()),json.loads(Path(a.candidate).read_text()))
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result.get('metrics',result),indent=2))
