from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
import math


alpha       = 0.6     
n_factors   = 50      
reg_all     = 0.02    
k_neighbors = 30     
threshold   = 4.0     


def compute_mae_rmse(preds):
    mae = sum(abs(est - true) for (_, _, true, est) in preds) / len(preds)
    mse = sum((est - true)**2       for (_, _, true, est) in preds) / len(preds)
    return mae, math.sqrt(mse)

def compute_precision(preds, th=threshold):
    tp = sum(1 for (_, _, true, est) in preds if est >= th and true >= th)
    fp = sum(1 for (_, _, true, est) in preds if est >= th and true <  th)
    return tp / (tp+fp) if (tp+fp)>0 else 0.0

def compute_coverage(preds, all_items, th=threshold):
    rec_items = {iid for (_, iid, _, est) in preds if est >= th}
    return len(rec_items) / all_items


def eval_fold(base_file, test_file):
    reader   = Reader(line_format='user item rating timestamp', sep='\t')
    data_train = Dataset.load_from_file(base_file, reader=reader).build_full_trainset()
    data_test  = Dataset.load_from_file(test_file, reader=reader).construct_testset(
                    Dataset.load_from_file(test_file, reader=reader).raw_ratings)


    algo_svd = SVD(n_factors=n_factors, reg_all=reg_all, random_state=42)
    algo_svd.fit(data_train)
    sim_options = {'name':'cosine','user_based':False}
    algo_knn = KNNBasic(k=k_neighbors, sim_options=sim_options)
    algo_knn.fit(data_train)


    preds_svd, preds_knn, preds_hyb = [], [], []
    for uid,iid,true,_ in data_test:
        est_svd = algo_svd.predict(uid,iid,r_ui=true).est
        est_knn = algo_knn.predict(uid,iid,r_ui=true).est
        est_hyb = alpha*est_svd + (1-alpha)*est_knn

        preds_svd.append((uid,iid,true,est_svd))
        preds_knn.append((uid,iid,true,est_knn))
        preds_hyb.append((uid,iid,true,est_hyb))


    mae_svd, rmse_svd = compute_mae_rmse(preds_svd)
    mae_knn, rmse_knn = compute_mae_rmse(preds_knn)
    mae_hy,  rmse_hy  = compute_mae_rmse(preds_hyb)

    prec_svd = compute_precision(preds_svd)
    prec_knn = compute_precision(preds_knn)
    prec_hy  = compute_precision(preds_hyb)

    cov_svd = compute_coverage(preds_svd, data_train.n_items)
    cov_knn = compute_coverage(preds_knn, data_train.n_items)
    cov_hy  = compute_coverage(preds_hyb,  data_train.n_items)

    return {
        'SVD':    (mae_svd, rmse_svd, prec_svd, cov_svd),
        'ItemKNN':(mae_knn, rmse_knn, prec_knn, cov_knn),
        'Hybrid': (mae_hy,  rmse_hy,  prec_hy,  cov_hy )
    }


folds = [('u1.base','u1.test'),
         ('u2.base','u2.test'),
         ('u3.base','u3.test'),
         ('u4.base','u4.test'),
         ('u5.base','u5.test')]

results = {}
for name,(b,t) in zip(['u1','u2','u3','u4','u5'], folds):
    results[name] = eval_fold(b, t)


for fold, res in results.items():
    print(f"{fold}:")
    for model,(mae,rmse,prec,cov) in res.items():
        print(f"  – {model:<7} MAE={mae:.4f}  RMSE={rmse:.4f}  Precision={prec:.4f}  Coverage={cov:.4f}")
    print()
