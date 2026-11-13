import csv
import json
import shutil
import uuid
from contextlib import contextmanager

@contextmanager
def test_directory():
    fixture_base=Path("artifacts/test-fixtures").resolve()
    root=fixture_base/uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield str(root)
    finally:
        if root.resolve().parent != fixture_base:
            raise RuntimeError("Fixture cleanup escaped its directory")
        shutil.rmtree(root)
        try:
            fixture_base.rmdir()
        except OSError:
            pass

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from ticket_router.api import create_app
from ticket_router.domain import LABELS,decision
from ticket_router.prepare import prepare
from ticket_router.evaluate import metrics,compare,evaluate
from ticket_router.integrations import week3_intake,week3_annotation,week4_case

class FixtureClassifier:
    def __init__(self,raw='O365'): self.raw=raw
    def predict(self,text):
        return {**decision(self.raw),'raw_output':self.raw,'model_id':'TEST-FIXTURE','latency_ms':1}

class Tests(unittest.TestCase):
    def test_auth_and_availability(self):
        with TestClient(create_app(token='test')) as c:
            self.assertEqual(c.get('/health').json()['status'],'not_ready')
            self.assertEqual(c.post('/v1/route',json={'ticket_id':'1','text':'mail broken'}).status_code,401)
            self.assertEqual(c.post('/v1/route',headers={'Authorization':'Bearer test'},json={'ticket_id':'1','text':'mail broken'}).status_code,503)
    def test_contract_and_review(self):
        for raw,expected in [('O365',False),('Support general',True),('O365 or Software',True)]:
            with TestClient(create_app(FixtureClassifier(raw),token='test')) as c:
                response=c.post('/v1/route',headers={'Authorization':'Bearer test'},json={'ticket_id':'1','text':'mail broken'})
                self.assertEqual(response.status_code,200)
                result=response.json(); self.assertEqual(result['needs_human_review'],expected)
                self.assertNotIn('raw_output',result)
                self.assertFalse(week3_annotation(result)['approval_granted'])
    def test_validation(self):
        with TestClient(create_app(FixtureClassifier(),token='test')) as c:
            for text in ['   ','x'*12001]:
                self.assertEqual(c.post('/v1/route',headers={'Authorization':'Bearer test'},json={'ticket_id':'1','text':text}).status_code,422)
    def test_split_and_conflicts(self):
        with test_directory() as tmp:
            root=Path(tmp); src=root/'tickets.csv'
            rows=[{'text':f'{label} example {i}','label':label} for label in LABELS for i in range(10)]
            def write():
                with src.open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=['text','label']); w.writeheader(); w.writerows(rows)
            rows.append(dict(rows[0])); write()
            a=prepare(src,root/'a','text','label'); b=prepare(src,root/'b','text','label')
            self.assertEqual(a,b); self.assertEqual(a['duplicates_removed'],1)
            train=json.loads((root/'a/train.json').read_text()); val=json.loads((root/'a/validation.json').read_text())
            self.assertFalse({r['id'] for r in train}&{r['id'] for r in val})
            self.assertEqual(len(val),14); self.assertEqual(len(train),56)
            rows.append({'text':rows[0]['text'],'label':'Software'}); write()
            with self.assertRaises(ValueError): prepare(src,root/'c','text','label')
    def test_invalid_counts_as_error(self):
        result=metrics([{'expected':'O365','label':None,'latency_ms':None},{'expected':'O365','label':'O365','latency_ms':2}])
        self.assertEqual(result['accuracy'],.5); self.assertEqual(result['invalid_count'],1)
        self.assertEqual(result['classification_report']['O365']['recall'],.5)
    def test_comparison_rejects_mismatch(self):
        a=evaluate(FixtureClassifier(),[{'id':'1','text':'mail','label':'O365'}],'fixture')
        b={**a,'dataset_fingerprint':'different'}
        with self.assertRaises(ValueError): compare(a,b)
    def test_adapters_keep_labels_out_of_input(self):
        case=week4_case({'id':'1','text':'mail','label':'O365'})
        self.assertNotIn('label',case['input'])
        self.assertEqual(week3_intake('1',{'title':'Email','logs':'original logs'})['text'],'Email\n\noriginal logs')

    def test_strict_evaluation_preserves_exception(self):
        class BrokenInference:
            def predict(self,text):
                raise AttributeError('diagnostic detail')
        rows=[{'id':'case-1','text':'mail','label':'O365'}]
        with self.assertRaises(RuntimeError) as caught:
            evaluate(BrokenInference(),rows,'exception-path-test',strict=True)
        self.assertIsInstance(caught.exception.__cause__,AttributeError)
        self.assertEqual(str(caught.exception.__cause__),'diagnostic detail')

    def test_comparison_rejects_failed_inference(self):
        report={'predictions':[{'id':'case-1','expected':'O365','label':None,'error':'AttributeError'}]}
        with self.assertRaisesRegex(ValueError,'incomplete evaluations'):
            compare(report,report)

if __name__=='__main__': unittest.main()
