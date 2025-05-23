from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
from math import sqrt
import os

alpha = 0.7          
n_factors = 50       
reg_all = 0.02       
threshold = 4.0      


item_cols = ['movie_id','title','release','video_release','url'] + list(range(19))
item_df = pd.read_csv('u.item', sep='|', encoding='latin-1',
                      names=item_cols, usecols=[0]+list(range(5,24)))
genre_vec = { str(r.movie_id): r.iloc[1:].values.astype(int)
              for _,r in item_df.iterrows() }


def predict_cbf(uid, iid, trainset):
    try:
        iu = trainset.to_inner_uid(uid)
        vi = genre_vec[iid]
    except:
        return trainset.global_mean
    sims, vals = [], []
    for j,r in trainset.ur[iu]:
        raw_j = trainset.to_raw_iid(j)
        vj = genre_vec.get(raw_j)
        if vj is None: continue
        denom = np.linalg.norm(vi)*np.linalg.norm(vj)
        s = np.dot(vi,vj)/denom if denom>0 else 0
        sims.append(s); vals.append(s*r)
    if not sims or abs(sum(map(abs,sims)))<1e-6:
        return trainset.global_mean
    return sum(vals)/sum(map(abs,sims))


def run_fold(base_file, test_file):

    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data_train = Dataset.load_from_file(base_file, reader=reader)
    data_test  = Dataset.load_from_file(test_file, reader=reader)
    trainset = data_train.build_full_trainset()
    testset  = data_test.construct_testset(data_test.raw_ratings)


    algo = SVD(n_factors=n_factors, reg_all=reg_all, random_state=0)
    algo.fit(trainset)


    preds = []
    tp_set = set()
    for uid,iid, true_r, _ in testset:
        est_svd = algo.predict(uid, iid, r_ui=true_r).est
        est_cbf = predict_cbf(uid, iid, trainset)
        est = alpha*est_svd + (1-alpha)*est_cbf
        preds.append((true_r, est))
        if est>=threshold:
            tp_set.add(iid)

    n = len(preds)
    se = sum((e-t)**2 for t,e in preds)
    ae = sum(abs(e-t)     for t,e in preds)
    rmse = sqrt(se/n)
    mae  = ae/n

    tp = sum(1 for t,e in preds if e>=threshold and t>=threshold)
    fp = sum(1 for t,e in preds if e>=threshold and t< threshold)
    precision = tp/(tp+fp) if tp+fp>0 else 0.0
    coverage  = len(tp_set)/trainset.n_items

    return mae, rmse, precision, coverage


results = {}
for i in range(1,6):
    base = f'u{i}.base'
    test = f'u{i}.test'
    if os.path.exists(base) and os.path.exists(test):
        results[f'u{i}'] = run_fold(base, test)
    else:
        results[f'u{i}'] = ('n/a','n/a','n/a','n/a')


print(f"{'Fold':<4}  {'MAE':>6}  {'RMSE':>6}  {'Prec':>6}  {'Cov':>6}")
for fold,(mae,rmse,prec,cov) in results.items():
    print(f"{fold:<4}  {mae:6.4f}  {rmse:6.4f}  {prec:6.4f}  {cov:6.4f}")
