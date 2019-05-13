import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_fats(path, norm=False):
    ts = pd.read_csv(path)
    ids = ts['ID'].values
    ts = ts.drop('ID',axis=1)
    
    # Drop columns with nans
    drop_cols = ts.columns[np.where(np.sum(ts.isnull()))[0]]
    ts = ts.drop(drop_cols, axis=1)
    
    if norm:
        ts = pd.DataFrame(MinMaxScaler().fit_transform(ts), columns=ts.columns)
    
    features = ts.values
    
    db = {}
    db['names'] = ids
    db['fats'] = features
    db['fats_keys'] = ts.columns.values
    
    return db

def get_spec(spec_path):
    spec = np.load(spec_path).item()
    spec['vae_keys'] = ['spec'+str(i) for i in range(spec['vae'].shape[-1])]
    return spec

def dict_gen(d, d_filter, d_skip = None):
    for key, val in d.items():
        if not np.isin(key, d_skip):
            val = val[d_filter]
        yield key, val

def x_match(ts_path, ts_key, spec_path, spec_key, keys_path_x, filter_l=None):
    ts = get_fats(ts_path)
    spec = get_spec(spec_path)
    
    # Filter labels
    # Filter ts, spec in x-match that have features
    x_keys = pd.read_csv(keys_path_x)
    if not filter_l is None:
        keep = x_keys['label'].isin(filter_l)
        x_keys = x_keys[keep]
    x_filter = x_keys['css_num_ID'].isin(ts[ts_key]) & x_keys['sloan_file'].isin(spec[spec_key])
    x_keys = x_keys[x_filter]
    x_keys = x_keys.drop_duplicates(subset='css_num_ID')
    ts_keys = x_keys['css_num_ID'].values
    spec_keys = x_keys['sloan_file'].values
    
    # Filter ts, spec in x-match
    ts_filter = list(map(lambda key: np.where(ts[ts_key]==key)[0][0], ts_keys))
    ts_new = dict(dict_gen(ts, ts_filter, ['fats_keys']))
    spec_filter = list(map(lambda key: np.where(spec[spec_key]==key)[0][0], spec_keys))
    spec_new = dict(dict_gen(spec, spec_filter, ['vae_keys']))
    
    # Check no duplicated time series or spectra
    assert np.unique(x_keys['css_num_ID']).shape[0] == x_keys.shape[0]
    assert np.unique(x_keys['sloan_file']).shape[0] == x_keys.shape[0]
    
    return ts_new, spec_new, x_keys

def build_db(ts_path, ts_key, spec_path, spec_key, keys_path_x, filter_l=None):
    ts, spec, keys = x_match(ts_path, ts_key, spec_path, spec_key, keys_path_x, filter_l)
    db = dict()
    db['fats'] = ts['fats']
    db['fats_keys'] = ts['fats_keys']
    db['ts_names'] = ts['names']
    db['vae'] = spec['vae']
    db['vae_keys'] = spec['vae_keys']
    db['spec_names'] = spec['names']
    db['labels'] = keys['label'].values
    
    # Normalize
    scaler = MinMaxScaler()
    db['fats'] = scaler.fit_transform(db['fats'])
    db['vae'] = scaler.fit_transform(db['vae'])
    
    assert db['fats'].shape[0] == db['vae'].shape[0] == db['labels'].shape[0]
    
    return db