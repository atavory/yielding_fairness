#!/usr/bin/env python3
"""Freeze the held public-auto candidate run with source, split and model provenance."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import public_group_candidates as c

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['beMTPL97', *[f'freMPL{i}' for i in range(1,10)], 'norauto', 'swmotorcycle']
RAW_COMMIT = '227fb56b8734bdb7c0327a41180e01d2ddaeaf26'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-repo', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--n', type=int, default=50000)
    args = parser.parse_args()
    assert args.steps == 800 and args.n == 50000, 'Held rerun settings are fixed'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.raw_repo,text=True).strip() == RAW_COMMIT
    output = ROOT/'outputs/public_auto_repro'
    output.mkdir(parents=True,exist_ok=True)
    provenance_path = output/'PROVENANCE.json'
    sources = [
        Path(__file__),
        ROOT/'fairness/public_group_candidates.py',
        ROOT/'fairness/public_w16_candidates_registry.py',
    ]
    provenance = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
                  'raw_repository':'https://github.com/dutangc/CASdatasets', 'raw_commit':RAW_COMMIT,
                  'command':[sys.executable,*sys.argv],
                  'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy']},
                  'package_freeze':sorted(
                      f"{dist.metadata['Name']}=={dist.version}"
                      for dist in importlib.metadata.distributions()
                      if dist.metadata.get('Name')
                  ),
                  'compute':{'device':'CPU','blas_threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}},
                  'config':{'datasets':DATASETS,'sampling_seed':42,'max_n':50000,'seeds':list(range(10)),
                            'steps':800,'target':.1,'net_lambda':1.,'split':'64/16/20 stratified by loss occurrence',
                            'score_family':'linear raw W-BCE','preprocessing_fit':'training rows only'},
                  'datasets':{},'output_sha256':{}}

    def save():
        provenance_path.write_text(json.dumps(provenance,indent=2,default=str)+'\n')

    save()
    try:
        for name in DATASETS:
            raw = args.raw_repo/'data'/(name+'.rda')
            loader = c.LOCAL_REGISTRY['belgian' if name=='beMTPL97' else 'auto']
            frame,numeric,categorical,label,value,loss,group,dataset,account = loader(str(raw),args.n,42)
            assert account is None
            # R factors arrive as pandas categoricals. The shared transformer
            # fills missing labels before encoding; object storage permits its
            # NA sentinel without adding a category learned from held-out rows.
            for column in categorical:
                frame[column] = frame[column].astype(object)
            info = {'dataset':dataset,'raw_sha256':sha(raw),'n':len(frame), 'numeric_features':numeric,
                    'categorical_features':categorical,
                    'categorical_storage':'R factors converted to object before training-only encoding',
                    'sample_frame_sha256':ahash(pd.util.hash_pandas_object(frame[numeric+categorical],index=True).values),
                    'sample_value_sha256':ahash(value),'sample_loss_sha256':ahash(loss),
                    'group_counts':{str(g):int(np.sum(group==g)) for g in np.unique(group)},'splits':[]}
            provenance['datasets'][name] = info
            indices = np.arange(len(frame),dtype=np.int64)
            archive = output/(name+'_candidates.csv.gz')
            with gzip.open(archive,'wt',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=[*c.FIELDS,'row_id','score_family'],lineterminator='\n')
                writer.writeheader()
                for seed in range(10):
                    outer,test = c.stratified_split(indices,label,.20,seed)
                    train,validation = c.stratified_split(outer,label,.20,10000+seed)
                    assert len(set(train)&set(validation)) == len(set(train)&set(test)) == len(set(validation)&set(test)) == 0
                    assert np.array_equal(np.sort(np.r_[train,validation,test]),indices)
                    xtr,xva,xte = c.transform(frame,numeric,categorical,train,validation,test)
                    wc,wb = c.fit_wbce(xtr,loss[train],value[train],500000+seed,args.steps)
                    vc,vb,vl,vs = c.fit_raw_mean_regression(xtr,value[train],args.steps,seed+1)
                    lc,lb,ll,ls = c.fit_raw_mean_regression(xtr,loss[train],args.steps,seed+2)
                    val_score = xva@wc+wb
                    val_a = np.maximum(c.predict_raw_mean(xva,vc,vb,vl,vs),1e-9)
                    val_b = c.predict_raw_mean(xva,lc,lb,ll,ls)
                    policies = c.policy_thresholds(val_score,group[validation],value[validation],loss[validation],.1)
                    criteria = c.criterion_thresholds(val_score,val_a,val_b,value[validation],.1,1.)
                    snapshot = {'train_ids':train,'validation_ids':validation,'test_ids':test,
                                'wbce_coef':wc,'wbce_intercept':np.array(wb),'value_coef':vc,'value_intercept':np.array(vb),
                                'value_location':np.array(vl),'value_scale':np.array(vs),
                                'loss_coef':lc,'loss_intercept':np.array(lb),
                                'loss_location':np.array(ll),'loss_scale':np.array(ls)}
                    for split,ids,design in [('validation',validation,xva),('test',test,xte)]:
                        score=design@wc+wb
                        a=np.maximum(c.predict_raw_mean(design,vc,vb,vl,vs),1e-9); b=c.predict_raw_mean(design,lc,lb,ll,ls)
                        assert np.isfinite(score).all() and np.isfinite(a).all() and np.isfinite(b).all()
                        selected={p:c.select_by_thresholds(score,group[ids],threshold) for p,threshold in policies.items()}
                        crit_scores=c.criterion_scores(score,a,b,1.)
                        objections={p:crit_scores[p]>=threshold for p,threshold in criteria.items()}
                        for key,array in [('score',score),('estimated_value',a),('estimated_loss',b),('value',value[ids]),('loss',loss[ids]),('group',group[ids].astype(str))]:
                            snapshot[split+'_'+key]=array
                        for j,row_id in enumerate(ids):
                            row={'dataset':dataset,'family':'natural','seed':seed,'target':'.1','sample_split':split,
                                 'group':str(group[row_id]),'value':format(float(value[row_id]),'.17g'),'loss':format(float(loss[row_id]),'.17g'),
                                 'score':format(float(score[j]),'.17g'),'ratio_score':format(float(score[j]),'.17g'),
                                 'estimated_value':format(float(a[j]),'.17g'),'estimated_loss':format(float(b[j]),'.17g'),
                                 'account':'','row_id':int(row_id),'score_family':'linear_raw_wbce'}
                            row.update({'selected_'+p:int(values[j]) for p,values in selected.items()})
                            row.update({'selected_criterion_'+p:int(values[j]) for p,values in objections.items()})
                            writer.writerow(row)
                    frozen=output/f'{name}_seed_{seed}.npz'
                    np.savez_compressed(frozen,**snapshot)
                    num=frame[numeric].apply(pd.to_numeric,errors='coerce')
                    medians=num.iloc[train].median();num=num.fillna(medians).fillna(0.)
                    info['splits'].append({'seed':seed,'train_n':len(train),'validation_n':len(validation),'test_n':len(test),
                                           'train_ids_sha256':ahash(train),'validation_ids_sha256':ahash(validation),'test_ids_sha256':ahash(test),
                                           'numeric_training_medians':medians.to_dict(),'numeric_training_means':num.iloc[train].mean().to_dict(),
                                           'numeric_training_stds':num.iloc[train].std(ddof=0).to_dict(),
                                           'categorical_training_levels':{key:sorted(frame.iloc[train][key].fillna('NA').astype(str).unique()) for key in categorical},
                                           'policy_thresholds':policies,'criterion_thresholds':criteria})
                    provenance['output_sha256'][str(frozen.relative_to(ROOT))]=sha(frozen)
                    save();print(f'{name} seed {seed} complete',flush=True)
            provenance['output_sha256'][str(archive.relative_to(ROOT))]=sha(archive)
            save()
        provenance['status']='complete'
    except Exception as error:
        provenance['status']='failed';provenance['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        provenance['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__ == '__main__':
    main()
