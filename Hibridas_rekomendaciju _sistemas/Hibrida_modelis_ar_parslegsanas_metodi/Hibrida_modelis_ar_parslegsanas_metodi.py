from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
from math import sqrt
import os


k_neighbors = 30     
sim_options = {'name':'cosine','user_based':False}
threshold_relevant = 4.0 


item_cols = ['movie_id','title','release','video_release','url'] + list(range(19))
item_df = pd.read_csv('u.item', sep='|', encoding='latin-1', names=item_cols, usecols=[0]+list(range(5,24)))
genre_vec = { str(r.movie_id): r.iloc[1:].values.astype(int) for _,r in item_df.iterrows() }

def predict_cbf(raw_uid, raw_iid, trainset):
    try:
        inner_uid = trainset.to_inner_uid(raw_uid)
        v_i = genre_vec[raw_iid]
    except:
        return trainset.global_mean
    ratings = trainset.ur[inner_uid]
    sims, vals = [], []
    for inner_j, ruj in ratings:
        raw_j = trainset.to_raw_iid(inner_j)
        v_j = genre_vec.get(raw_j)
        if v_j is None: continue
        denom = np.linalg.norm(v_i)*np.linalg.norm(v_j)
        s = np.dot(v_i,v_j)/denom if denom>0 else 0
        sims.append(s); vals.append(s*ruj)
    if not sims or sum(map(abs,sims))<1e-6:
        return trainset.global_mean
    return sum(vals)/sum(map(abs,sims))


def run_fold(base_file, test_file):

    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data_train = Dataset.load_from_file(base_file, reader=reader)
    data_test  = Dataset.load_from_file(test_file, reader=reader)
    trainset = data_train.build_full_trainset()
    testset  = data_test.construct_testset(data_test.raw_ratings)

    algo_knn = KNNBasic(k=k_neighbors, sim_options=sim_options)
    algo_knn.fit(trainset)

    preds, tp= [],0
    rec_items = set()
    for uid,iid,true_r,_ in testset:
        inner_i = trainset.to_inner_iid(iid)
        neighbors = algo_knn.get_neighbors(inner_i, k_neighbors)
        if len(neighbors) < k_neighbors:
            est = predict_cbf(uid,iid,trainset)
        else:
            est = algo_knn.predict(uid,iid,r_ui=true_r).est

        preds.append((true_r, est))
        if est >= threshold_relevant:
            rec_items.add(iid)
            if true_r >= threshold_relevant:
                tp+=1

    n = len(preds)
    se = sum((e-t)**2 for t,e in preds)
    ae = sum(abs(e-t)     for t,e in preds)
    rmse = sqrt(se/n)
    mae  = ae/n


    fp = sum(1 for t,e in preds if e>=threshold_relevant and t < threshold_relevant)
    precision = tp/(tp+fp) if tp+fp>0 else 0.0


    coverage = len(rec_items)/trainset.n_items

    return mae, rmse, precision, coverage


results = {}
for i in range(1,6):
    base = f'u{i}.base'
    test = f'u{i}.test'
    if os.path.exists(base) and os.path.exists(test):
        results[f'u{i}'] = run_fold(base, test)
    else:
        results[f'u{i}'] = ('no data','no data','no data','no data')

print(f"{'Fold':<4}  {'MAE':>6}  {'RMSE':>6}  {'Precision':>9}  {'Coverage':>8}")
for fold,(mae,rmse,prec,cov) in results.items():
    print(f"{fold:<4}  {mae:6.4f}  {rmse:6.4f}  {prec:9.4f}  {cov:8.4f}")
