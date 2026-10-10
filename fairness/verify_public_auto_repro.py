#!/usr/bin/env python3
"""Audit frozen public-auto inputs, split IDs, transforms, models and CSV flags."""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import public_group_candidates as c

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-repo',type=Path,required=True)
    args=parser.parse_args()
    output=ROOT/'outputs/public_auto_repro'
    provenance_path=output/'PROVENANCE.json'
    provenance=json.loads(provenance_path.read_text())
    assert provenance['status']=='complete'
    assert provenance['config']['seeds']==list(range(10))
    for name,expected in provenance['source_sha256'].items():
        frozen_code=subprocess.check_output(['git','show',provenance['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(frozen_code).hexdigest()==expected,name
    hashes={str(provenance_path.relative_to(ROOT)):sha(provenance_path)}
    for name,expected in provenance['output_sha256'].items():
        actual=sha(ROOT/name);assert actual==expected,name;hashes[name]=actual
    maximum_score_error=0.;csv_rows=0
    for name,info in provenance['datasets'].items():
        raw=args.raw_repo/'data'/(name+'.rda');assert sha(raw)==info['raw_sha256']
        loader=c.LOCAL_REGISTRY['belgian' if name=='beMTPL97' else 'auto']
        frame,numeric,categorical,label,value,loss,group,dataset,account=loader(str(raw),50000,42)
        for column in categorical: frame[column]=frame[column].astype(object)
        assert len(frame)==info['n'] and dataset==info['dataset']
        assert ahash(pd.util.hash_pandas_object(frame[numeric+categorical],index=True).values)==info['sample_frame_sha256']
        assert ahash(value)==info['sample_value_sha256'] and ahash(loss)==info['sample_loss_sha256']
        assert len(info['splits'])==10
        frozen_by_cell={}
        for split_info in info['splits']:
            seed=split_info['seed']
            with np.load(output/f'{name}_seed_{seed}.npz') as archive:
                frozen={key:archive[key] for key in archive.files}
            train,validation,test=[frozen[s+'_ids'] for s in ['train','validation','test']]
            assert np.array_equal(np.sort(np.r_[train,validation,test]),np.arange(len(frame)))
            assert not np.intersect1d(train,validation).size and not np.intersect1d(train,test).size and not np.intersect1d(validation,test).size
            for split,ids in [('train',train),('validation',validation),('test',test)]:
                assert len(ids)==split_info[split+'_n'] and ahash(ids)==split_info[split+'_ids_sha256']
            numbers=frame[numeric].apply(pd.to_numeric,errors='coerce')
            medians=numbers.iloc[train].median()
            filled=numbers.fillna(medians).fillna(0.)
            mean=filled.iloc[train].mean();std=filled.iloc[train].std(ddof=0)
            for actual,key in [(medians,'numeric_training_medians'),(mean,'numeric_training_means'),(std,'numeric_training_stds')]:
                assert np.allclose(actual.values,[split_info[key][k] for k in numeric],atol=1e-10,rtol=1e-12)
            scales=std.to_numpy().copy();scales[scales<=0]=1.
            for key in categorical:
                assert sorted(frame.iloc[train][key].fillna('NA').astype(str).unique())==split_info['categorical_training_levels'][key]
            for split,ids in [('validation',validation),('test',test)]:
                parts=[(filled.iloc[ids].to_numpy(dtype=np.float64)-mean.to_numpy(dtype=np.float64))/scales]
                for key in categorical:
                    labels=frame.iloc[ids][key].fillna('NA').astype(str).to_numpy()
                    levels=split_info['categorical_training_levels'][key]
                    parts.append(np.column_stack([labels==level for level in levels]).astype(float))
                design=np.concatenate(parts,axis=1)
                score=design@frozen['wbce_coef']+frozen['wbce_intercept']
                maximum_score_error=max(maximum_score_error,float(np.max(np.abs(score-frozen[split+'_score']))))
                assert np.allclose(score,frozen[split+'_score'],atol=1e-8,rtol=1e-10)
                for prefix in ['value','loss']:
                    predicted=c.predict_raw_mean(
                        design,
                        frozen[prefix+'_coef'],
                        float(frozen[prefix+'_intercept']),
                        float(frozen[prefix+'_location']),
                        float(frozen[prefix+'_scale']),
                    )
                    if prefix=='value': predicted=np.maximum(predicted,1e-9)
                    assert np.allclose(predicted,frozen[split+'_estimated_'+prefix],atol=1e-8,rtol=1e-10)
                assert np.array_equal(frozen[split+'_value'],value[ids]) and np.array_equal(frozen[split+'_loss'],loss[ids])
                assert np.array_equal(frozen[split+'_group'],group[ids].astype(str))
                frozen_by_cell[seed,split]=(frozen,{int(row_id):i for i,row_id in enumerate(ids)},split_info)
        seen=set()
        with gzip.open(output/(name+'_candidates.csv.gz'),'rt') as stream:
            for row in csv.DictReader(stream):
                seed,split,row_id=int(row['seed']),row['sample_split'],int(row['row_id'])
                key=(seed,split,row_id);assert key not in seen;seen.add(key)
                frozen,positions,split_info=frozen_by_cell[seed,split];j=positions[row_id]
                assert row['dataset']==dataset and row['score_family']=='linear_raw_wbce'
                for column in ['score','estimated_value','estimated_loss','value','loss']:
                    assert float(row[column])==float(frozen[split+'_'+column][j]),(name,key,column)
                assert row['group']==str(frozen[split+'_group'][j])
                score=float(row['score'])
                for policy,thresholds in split_info['policy_thresholds'].items():
                    assert int(row['selected_'+policy])==int(score>=float(thresholds[row['group']]))
                crit={'wbce_ratio':score,'harm':float(row['estimated_loss']),
                      'net_harm':float(row['estimated_loss'])-float(row['estimated_value'])}
                for criterion,threshold in split_info['criterion_thresholds'].items():
                    assert int(row['selected_criterion_'+criterion])==int(crit[criterion]>=float(threshold))
                csv_rows+=1
        assert len(seen)==sum(s['validation_n']+s['test_n'] for s in info['splits'])
        print(name,'PASS',len(seen),'candidate rows',flush=True)
    report={'status':'PASS','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'verifier_sha256':sha(Path(__file__)),'input_sha256':hashes,'datasets':len(provenance['datasets']),
            'splits':120,'fitted_models':360,'candidate_rows':csv_rows,'maximum_score_replay_error':maximum_score_error,
            'checks':['source and raw hashes','sample fingerprint','disjoint complete splits','training-only transformation parameters',
                      'predictions replayed from coefficients','raw outcome identity','CSV identities and validation-frozen policy flags']}
    (output/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
