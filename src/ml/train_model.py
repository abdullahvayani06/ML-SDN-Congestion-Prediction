#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)

def generate_episode(seed, n_steps=600, primary_cap=20.0):
    rng = np.random.RandomState(seed)
    tx_mbps = np.zeros(n_steps)
    idx = 0
    while idx < n_steps:
        bg_len = rng.randint(6, 12)
        for _ in range(bg_len):
            if idx < n_steps:
                tx_mbps[idx] = rng.uniform(6.2, 7.2)
                idx += 1
        if idx < n_steps:
            tx_mbps[idx] = rng.uniform(9.4, 13.4)
            idx += 1
        if idx < n_steps and rng.rand() < 0.5:
            tx_mbps[idx] = rng.uniform(12.0, 13.8)
            idx += 1
        burst_len = rng.randint(10, 18)
        for _ in range(burst_len):
            if idx < n_steps:
                tx_mbps[idx] = rng.uniform(15.8, 19.8)
                idx += 1
        if idx < n_steps:
            tx_mbps[idx] = rng.uniform(9.0, 11.5)
            idx += 1

    rx_mbps = np.clip(tx_mbps * rng.uniform(0.97, 1.02, n_steps), 0.5, primary_cap)
    s_tx = pd.Series(tx_mbps)
    util_pct = np.clip((s_tx.values / primary_cap) * 100.0, 0.0, 100.0)
    delta_mbps = s_tx.diff().fillna(0.0).values
    roll_mean = s_tx.rolling(window=3, min_periods=1).mean().fillna(0.0).values
    roll_std = s_tx.rolling(window=3, min_periods=1).std().fillna(0.0).values

    u_next1 = pd.Series(util_pct).shift(-1).ffill().bfill().values
    u_next2 = pd.Series(util_pct).shift(-2).ffill().bfill().values
    future_util_pct = np.maximum(u_next1, u_next2)
    future_congested = (future_util_pct >= 70.0).astype(int)

    return pd.DataFrame({
        'rx_mbps': rx_mbps,
        'tx_mbps': s_tx.values,
        'utilization_pct': util_pct,
        'delta_mbps': delta_mbps,
        'rolling_mean_mbps': roll_mean,
        'rolling_std_mbps': roll_std,
        'future_util_pct': future_util_pct,
        'future_congested': future_congested
    })

def main():
    train_df = pd.concat([generate_episode(101, 600), generate_episode(102, 600), generate_episode(103, 600)], ignore_index=True)
    test_df = pd.concat([generate_episode(201, 400), generate_episode(202, 400)], ignore_index=True)
    feature_cols = ['rx_mbps', 'tx_mbps', 'utilization_pct', 'delta_mbps', 'rolling_mean_mbps', 'rolling_std_mbps']

    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(
        train_df[feature_cols].values, train_df['future_congested'].values
    )
    gbr_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42).fit(
        train_df[feature_cols].values, train_df['future_util_pct'].values
    )

    y_cls_pred = rf_cls.predict(test_df[feature_cols].values)
    y_reg_pred = gbr_reg.predict(test_df[feature_cols].values)

    acc = accuracy_score(test_df['future_congested'], y_cls_pred) * 100.0
    prec = precision_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    rec = recall_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    f1 = f1_score(test_df['future_congested'], y_cls_pred, zero_division=0) * 100.0
    rmse_norm = np.sqrt(mean_squared_error(test_df['future_util_pct'] / 100.0, y_reg_pred / 100.0))
    mae_norm = mean_absolute_error(test_df['future_util_pct'] / 100.0, y_reg_pred / 100.0)

    print("=================================================================")
    print(" UNSEEN SYNTHETIC EPISODES VALIDATION (800 SAMPLES, NO LEAKAGE)  ")
    print("=================================================================")
    print(f"Classification Accuracy  : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")
    print(f"Normalized Forecast RMSE : {rmse_norm:.4f} | Normalized MAE: {mae_norm:.4f}")
    print("=================================================================")

    with open('sdn_congestion_rf.pkl', 'wb') as f:
        pickle.dump({'rf_cls': rf_cls, 'gbr_reg': gbr_reg, 'features': feature_cols}, f)

if __name__ == '__main__':
    main()
