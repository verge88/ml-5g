import json
import numpy as np
from paper5g.prepare import features,partition,build_dataset

def rows(n=100):
    return [dict(time=float(i),source='a',destination='b',method='GET',path='/service/v1/resource',wire='HTTP2',label=0,kind='normal') for i in range(n)]

def test_features_are_independent_of_labels_markers_and_future_rows():
    original=rows();changed=[dict(r,label=1,kind='secret',episode='private') for r in original]
    one=features(original,{'a':1,'b':2});two=features(changed,{'a':1,'b':2})
    for a,b in zip(one,two):np.testing.assert_array_equal(a,b)
    future=features(original+[dict(original[-1],time=200,destination='new')],{'a':1,'b':2})
    for a,b in zip(one,future):np.testing.assert_array_equal(a,b[:len(a)])

def test_timestamp_groups_are_not_shared_across_partitions():
    data=rows()
    for i in range(35,45):data[i]['time']=39.
    parts=partition(data)
    assert sum(map(len,parts.values()))==len(data)
    assert max(r['time'] for r in parts['train'])<min(r['time'] for r in parts['calibration'])
    assert max(r['time'] for r in parts['calibration'])<min(r['time'] for r in parts['clean_holdout'])

def test_full_test_rows_unknown_labels_and_train_only_vocabulary(tmp_path):
    normal=rows();target=[dict(r,destination='unseen',label=1,kind='example') for r in rows(123)]
    target[-1]['label']=-1
    info=build_dataset('fixture',normal,{'mixed':target},tmp_path)
    assert info['n_nodes']==3
    assert info['streams']['mixed']['rows']==123 and info['streams']['mixed']['unknown']==1
    data=np.load(tmp_path/'fixture/mixed.npz',allow_pickle=False)
    assert len(data['x'])==123 and np.all(data['destination']==0)
    assert info['streams']['train']['rows']==40 and info['streams']['clean_holdout']['rows']==50
    assert not any(k in data.files for k in ('ip','path','marker','checksum'))
