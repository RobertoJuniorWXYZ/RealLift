import pandas as pd
import numpy as np
import cvxpy as cp
from reallift.geo.validation import validate_geo_clusters

def _run_oof_refinement_single(cluster, filepath, date_col, df_pre, start_date, end_date, n_folds, experiment_days=None, experiment_type="synthetic_control", df=None):
    """
    Iterative pruning and Out-of-Fold (OOF) refinement of a single cluster.
    
    This function progressively removes the least significant donors from the synthetic 
    design while monitoring cross-validation metrics (R², MAPE, WAPE). It identifies 
    the 'Goldilocks' point where the model is complex enough to capture behavior but 
    simple enough to avoid overfitting.
    """
    current_cluster = cluster.copy()
    history = []

    while True:
        if experiment_type == "matched_did":
            # Bypass OOF splitting for Matched DiD, evaluate holistic equal-weighted R² 
            t_cols = current_cluster["treatment"]
            c_cols = current_cluster["control"]
            y_hol = df_pre[t_cols].mean(axis=1).values.astype(float)
            if len(c_cols) > 0:
                X_hol = df_pre[c_cols].mean(axis=1).values.astype(float)
            else:
                X_hol = np.zeros_like(y_hol)
                
            if np.std(y_hol) > 0 and np.std(X_hol) > 0:
                corr_hol = float(np.corrcoef(y_hol, X_hol)[0, 1])
            else:
                corr_hol = 0.0
                
            # Treat r2_holistic identically so OOF rules just validate raw correlation
            r2_hol = corr_hol ** 2
            
            cv_row = pd.Series({
                "treatment": current_cluster["treatment"],
                "r2_train": r2_hol,
                "r2_test": r2_hol,
                "mape_train": 0.0, "mape_test": 0.0,
                "wape_train": 0.0, "wape_test": 0.0
            })
        else:
            validation = validate_geo_clusters(
                filepath=filepath,
                date_col=date_col,
                splits=[current_cluster],
                treatment_start_date=end_date,
                start_date=start_date,
                n_folds=n_folds,
                test_size=experiment_days,
                plot=False,
                verbose=False,
                df=df
            )
            cv_row = validation["summary"].iloc[0]
            
        history.append((current_cluster.copy(), cv_row))

        controls = current_cluster["control"].copy()
        weights = current_cluster.get("control_weights", []).copy()
        
        if len(controls) <= 1:
            break
        
        min_idx = int(np.argmin(weights)) if len(weights) > 0 else -1
        if min_idx >= 0:
            controls.pop(min_idx)
            weights.pop(min_idx)
        else:
            controls.pop()
        
        t_cols = current_cluster["treatment"]
        y_syn = df_pre[t_cols].mean(axis=1).values.astype(float)
        X_syn = df_pre[controls].values.astype(float)
        
        y_mean_syn = y_syn.mean() if y_syn.mean() != 0 else 1e-10
        X_mean_syn = X_syn.mean(axis=0)
        X_mean_syn[X_mean_syn == 0] = 1e-10

        y_norm_syn = y_syn / y_mean_syn
        X_norm_syn = X_syn / X_mean_syn

        w_syn = cp.Variable(len(controls))
        
        # NO INTERCEPT in refinement to prevent leakage absorption
        obj_syn = cp.Minimize(cp.sum_squares(y_norm_syn - (X_norm_syn @ w_syn)))
        cons_syn = [w_syn >= 0, cp.sum(w_syn) == 1]
        prob_syn = cp.Problem(obj_syn, cons_syn)
        
        try:
            prob_syn.solve(solver=cp.SCS, verbose=False)
            w_vals = np.array(w_syn.value).flatten()
            w_vals[w_vals < 0] = 0.0
            sum_w = np.sum(w_vals)
            if sum_w > 0:
                w_vals = w_vals / sum_w
            new_weights = [float(w) for w in w_vals]
            
            # NO INTERCEPT in refinement models
            alpha_val = 0.0 
            y_pred = (X_norm_syn @ np.array(new_weights)) * y_mean_syn
            
            if np.std(y_syn) > 0 and np.std(y_pred) > 0:
                corr = float(np.corrcoef(y_syn, y_pred)[0, 1])
            else:
                corr = 0.0
        except Exception:
            new_weights = [1.0/len(controls)] * len(controls)
            corr = 0.0

        current_cluster["control"] = controls
        current_cluster["control_weights"] = new_weights
        current_cluster["correlation"] = corr

    valid_steps = [
        step for step in history 
        if step[1]["r2_test"] >= 0.6 
        and step[1]["r2_train"] >= 0.6 
        and abs(step[1]["r2_train"] - step[1]["r2_test"]) <= 0.20
    ]

    if valid_steps:
        valid_steps.sort(key=lambda x: x[1]["r2_test"], reverse=True)
        best_cluster, best_cv_row = valid_steps[0]
        passed = True
    else:
        history.sort(key=lambda x: x[1]["r2_test"], reverse=True)
        best_cluster, best_cv_row = history[0]
        passed = False
        
    if experiment_type == "matched_did":
        controls = best_cluster["control"]
        if len(controls) > 0:
            best_cluster["control_weights"] = [1.0 / len(controls)] * len(controls)
        
    return best_cluster, best_cv_row, passed, len(history)
