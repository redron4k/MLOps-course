import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import release


def fixtures():
    runtime={'base_revision':'fixed-revision','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},
             'dtype':'float32'}
    digest=hashlib.sha256(json.dumps(runtime,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    manifest={'schema_version':1,'model_id':'training-run-A','model_sha256':'c'*64,
              'protocol_sha256':'d'*64,'golden_sha256':'e'*64,'runtime_sha256':digest,
              'runtime':runtime,'artifact_uri':'models:/my-model/1'}
    report={'status':'complete','passed':True,'evaluation_run_id':'eval-A',
            'model_id':manifest['model_id'],**{k:manifest[k] for k in ['model_sha256','protocol_sha256','golden_sha256','runtime_sha256']},
            'metrics':{'accuracy':.8,'format_rate':1.0}}
    return manifest,report


class ReleaseContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'input.jsonl';self.data.write_text('пример\n')
        self.config={'tokenizer_revision':'r1','template_sha256':'a'*64,'max_length':128,'code_revision':'c1'}
        self.manifest,self.report=fixtures()
        self.target=self.root/'accepted.json';self.target.write_text('{"previous": true}\n')
        self.before=self.target.read_bytes()

    def test_key_ignores_path_and_mapping_order(self):
        other=self.root/'other.jsonl';other.write_bytes(self.data.read_bytes())
        key=release.cache_key(self.data,self.config)
        self.assertRegex(key,r'^[0-9a-f]{64}$')
        self.assertEqual(key,release.cache_key(other,dict(reversed(list(self.config.items())))))

    def test_changed_bytes_at_same_path_invalidate_cache(self):
        old=release.cache_key(self.data,self.config);self.data.write_text('другие данные\n')
        self.assertNotEqual(old,release.cache_key(self.data,self.config))

    def test_every_preparation_setting_invalidates_cache(self):
        old=release.cache_key(self.data,self.config)
        for key,value in [('tokenizer_revision','r2'),('template_sha256','f'*64),('max_length',256),('code_revision','c2')]:
            with self.subTest(key=key):
                config={**self.config,key:value};self.assertNotEqual(old,release.cache_key(self.data,config))

    def test_complete_bound_report_allowed(self):
        self.assertIsNone(release.validate_release(self.manifest,self.report))

    def test_incomplete_false_and_truthy_string_rejected(self):
        for key,value in [('status','failed'),('status','running'),('passed',False),('passed','true'),('evaluation_run_id','')]:
            with self.subTest(key=key,value=value):
                bad={**self.report,key:value}
                with self.assertRaises(ValueError):release.validate_release(self.manifest,bad)

    def test_missing_fields_rejected(self):
        for side,keys in [('manifest',list(self.manifest)),('report',list(self.report))]:
            for key in keys:
                with self.subTest(side=side,key=key):
                    m,r=copy.deepcopy(self.manifest),copy.deepcopy(self.report)
                    del (m if side=='manifest' else r)[key]
                    with self.assertRaises(ValueError):release.validate_release(m,r)

    def test_report_binding_mismatch_rejected(self):
        for key in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']:
            with self.subTest(key=key):
                bad={**self.report,key:('other-model' if key=='model_id' else 'f'*64)}
                with self.assertRaises(ValueError):release.validate_release(self.manifest,bad)

    def test_runtime_change_with_stale_hash_rejected(self):
        for key,value in [('dtype','float16'),('template_sha256','f'*64),('generation',{'max_new_tokens':64,'do_sample':False})]:
            with self.subTest(key=key):
                m=copy.deepcopy(self.manifest);m['runtime'][key]=value
                with self.assertRaises(ValueError):release.validate_release(m,self.report)

    def test_invalid_metric_rejected(self):
        for value in [float('nan'),float('inf'),-.1,1.1,True,'0.8',None]:
            for metric in ['accuracy','format_rate']:
                with self.subTest(value=value,metric=metric):
                    r=copy.deepcopy(self.report);r['metrics'][metric]=value
                    with self.assertRaises(ValueError):release.validate_release(self.manifest,r)

    def test_invalid_digest_and_schema_rejected(self):
        for key,value in [('schema_version',True),('schema_version',2),('model_sha256','latest'),('model_id',''),('artifact_uri','')]:
            with self.subTest(key=key):
                m={**self.manifest,key:value}
                with self.assertRaises(ValueError):release.validate_release(m,self.report)

    def test_publish_and_repeat(self):
        release.publish(self.manifest,self.report,self.target)
        result=self.target.read_bytes();self.assertEqual(json.loads(result),self.manifest)
        release.publish(self.manifest,self.report,self.target)
        self.assertEqual(self.target.read_bytes(),result)

    def test_rejection_preserves_previous_bytes(self):
        bad={**self.report,'passed':False}
        with self.assertRaises(ValueError):release.publish(self.manifest,bad,self.target)
        self.assertEqual(self.target.read_bytes(),self.before)

    def test_failed_atomic_replace_preserves_previous_bytes(self):
        with patch('os.replace',side_effect=OSError('simulated disk error')):
            with self.assertRaises(OSError):release.publish(self.manifest,self.report,self.target)
        self.assertEqual(self.target.read_bytes(),self.before)
        self.assertEqual(list(self.root.glob('*.tmp')),[])


if __name__=='__main__':unittest.main(verbosity=2)
