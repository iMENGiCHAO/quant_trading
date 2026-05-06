#!/usr/bin/env python3
"""
Hyperion Quant — A 股真实数据分析和市场预测报告
=====================================================
数据源: Yahoo Finance (通过 Clash 代理)
"""

import sys, os, json, warnings
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 设置代理 (Clash)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


# ==================== 数据获取 ====================

def fetch_data_yahoo():
    """从 Yahoo Finance 获取 A 股数据"""
    import requests, urllib3
    urllib3.disable_warnings()

    # A 股核心股票 (沪深300精选)
    stocks = [
        ("600519.SS", "贵州茅台"), ("000858.SZ", "五粮液"), ("000568.SZ", "泸州老窖"),
        ("600036.SS", "招商银行"), ("601318.SS", "中国平安"), ("600030.SS", "中信证券"),
        ("300750.SZ", "宁德时代"), ("000333.SZ", "美的集团"), ("000651.SZ", "格力电器"),
        ("600900.SS", "长江电力"), ("601166.SS", "兴业银行"), ("600585.SS", "海螺水泥"),
        ("002415.SZ", "海康威视"), ("000002.SZ", "万科A"), ("600276.SS", "恒瑞医药"),
        ("601012.SS", "隆基绿能"), ("600309.SS", "万华化学"), ("002475.SZ", "立讯精密"),
        ("000001.SZ", "平安银行"), ("600887.SS", "伊利股份"),
        ("002714.SZ", "牧原股份"), ("300059.SZ", "东方财富"),
        ("600809.SS", "山西汾酒"), ("601899.SS", "紫金矿业"),
        ("600690.SS", "海尔智家"), ("000725.SZ", "京东方A"),
        ("002352.SZ", "顺丰控股"), ("603259.SS", "药明康德"),
        ("600030.SS", "中信证券"), ("601688.SS", "华泰证券"),
    ]
    # 去重
    seen = set()
    stocks = [(s,n) for s,n in stocks if not (s in seen or seen.add(s))]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    print(f"[1/5] 获取数据 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})...")

    all_data = []
    errors = []
    session = requests.Session()
    session.verify = False

    for i, (symbol, name) in enumerate(stocks):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'range': '2y',
                'interval': '1d',
                'includePrePost': 'false',
            }
            resp = session.get(url, params=params,
                              headers={'User-Agent': 'Mozilla/5.0'},
                              timeout=15)
            if resp.status_code != 200:
                errors.append(f"{symbol}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            result = data.get('chart', {}).get('result', [])
            if not result:
                errors.append(f"{symbol}: no data")
                continue

            quotes = result[0]
            timestamps = quotes.get('timestamp', [])
            indicators = quotes.get('indicators', {})
            quote_list = indicators.get('quote', [])
            if not quote_list:
                errors.append(f"{symbol}: no quote data")
                continue
            quotes_data = quote_list[0]
            adjclose_list = indicators.get('adjclose', [{}])[0].get('adjclose', [])
            raw_close = quotes_data.get('close', [])

            # 优先用复权价，否则用收盘价
            close_series = adjclose_list if len(adjclose_list) == len(timestamps) else raw_close

            if len(timestamps) < 20:
                errors.append(f"{symbol}: only {len(timestamps)} days")
                continue

            df = pd.DataFrame({
                'date': pd.to_datetime(timestamps, unit='s'),
                'open': quotes_data.get('open', []),
                'high': quotes_data.get('high', []),
                'low': quotes_data.get('low', []),
                'close': close_series,
                'volume': quotes_data.get('volume', []),
            })
            df['instrument'] = symbol.replace('.SS', '').replace('.SZ', '')
            df['name'] = name
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            df = df.dropna(subset=['open', 'close', 'high', 'low'])
            all_data.append(df)

        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:40]}")

        if (i+1) % 5 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")

    if not all_data:
        print("  ERROR: 无数据!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['date', 'instrument']).reset_index(drop=True)
    print(f"  OK: {len(all_data)} 只股票, {len(combined)} 行")
    if errors:
        print(f"  失败: {len(errors)} 只 (如 {errors[0][:40]})")
    return combined


# ==================== 因子 + 标签 ====================

def compute_features(data):
    """生成因子和标签"""
    print("\n[2/5] 计算因子和标签...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hyperion.alpha.factors import Alpha158

    alpha = Alpha158()
    all_features = []

    for (code, name), grp in data.groupby(['instrument', 'name']):
        grp = grp.sort_values('date').reset_index(drop=True)
        ohlcv = grp[['open', 'high', 'low', 'close', 'volume', 'vwap']]
        try:
            factors = alpha.extract(ohlcv)
            factors.columns = [f"f_{c}" for c in factors.columns]
            factors['instrument'] = code
            factors['name'] = name
            factors['date'] = grp['date'].values
            # 未来5日收益
            factors['label'] = grp['close'].shift(-5) / grp['close'] - 1
            all_features.append(factors)
        except Exception as e:
            print(f"  WARN: {code} factor err: {str(e)[:30]}")

    result = pd.concat(all_features, ignore_index=True)
    print(f"  因子: {result.shape[1]-4} 个, {len(result)} 样本")
    return result


# ==================== 训练 ====================

def train_lightgbm(df):
    """训练 LightGBM"""
    print("\n[3/5] 训练 LightGBM...")
    import lightgbm as lgb

    feat_cols = [c for c in df.columns if c.startswith('f_')]
    df_model = df.dropna(subset=['label'] + feat_cols[:1])
    X = df_model[feat_cols].fillna(0).values
    y = df_model['label'].values

    n = len(X)
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

    model = lgb.LGBMRegressor(
        n_estimators=500, num_leaves=15, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=0.3,
        min_child_samples=50, min_child_weight=5.0,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

    preds = model.predict(X_test)
    imp = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"  最佳迭代: {model.best_iteration_}")
    return model, X_train, X_test, y_train, y_test, preds, imp, df_model, feat_cols


# ==================== 评估 ====================

def evaluate(y_test, preds):
    print("\n[4/5] 评估...")
    valid = ~np.isnan(y_test) & ~np.isnan(preds)
    yv, pv = y_test[valid], preds[valid]
    if len(yv) < 10:
        return 0, pd.Series(), 0

    rank_ic = np.corrcoef(pd.Series(yv).rank(), pd.Series(pv).rank())[0, 1]

    if len(yv) >= 20:
        groups = pd.qcut(pd.Series(pv), 5, labels=['Q1低','Q2','Q3','Q4','Q5高'])
        decile = pd.DataFrame({'group': groups, 'ret': yv}).groupby('group')['ret'].mean()
        spread = decile.iloc[-1] - decile.iloc[0]
    else:
        decile = pd.Series()
        spread = 0

    print(f"  Rank IC: {rank_ic:.4f}")
    print(f"  多空收益差: {spread:.2%}" if spread != 0 else "  多空收益差: N/A")
    return rank_ic, decile, spread


# ==================== 预测 ====================

def predict_and_report(df, model, feat_cols, data, rank_ic, decile, spread):
    print("\n[5/5] 生成预测...")
    feat_cols_all = [c for c in df.columns if c.startswith('f_')]
    X_all = df[feat_cols_all].fillna(0).values
    preds_all = model.predict(X_all)
    df = df.copy()
    df['pred_ret'] = np.nan
    df.iloc[-len(preds_all):, df.columns.get_loc('pred_ret')] = preds_all

    # merge 收盘价
    price_info = data.groupby(['instrument','name']).last()[['close']].reset_index()
    price_info['instrument'] = price_info['instrument'].astype(str)
    price_info['name'] = price_info['name'].astype(str)

    # 最新一期预测
    latest = df.dropna(subset=['pred_ret']).groupby(['instrument','name']).last().sort_values('pred_ret', ascending=False)
    latest = latest.reset_index()
    latest['instrument'] = latest['instrument'].astype(str)
    latest['name'] = latest['name'].astype(str)
    latest = latest.merge(price_info, on=['instrument','name'], how='left')
    latest = latest.set_index(['instrument','name'])

    top5 = latest.head(5)
    bot5 = latest.tail(5)

    # 报告
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    total_market_ret = data.groupby('date')['close'].mean().pct_change().sum()

    print("\n" + "="*70)
    print("  HYPERION QUANT — A 股市场分析报告")
    print("="*70)
    print(f"  生成时间: {now}")
    print(f"  数据区间: {data['date'].min().strftime('%Y-%m-%d')} ~ {data['date'].max().strftime('%Y-%m-%d')}")
    print(f"  股票: {data['instrument'].nunique()} 只")
    print(f"  样本: {len(data):,} 行")
    print(f"\n{'─'*70}")
    print("  一、模型表现")
    print(f"{'─'*70}")
    print(f"  模型: LightGBM (200 trees, 31 leaves)")
    print(f"  Rank IC: {rank_ic:.4f}")
    if spread != 0:
        print(f"  多空收益差: {spread:.2%}")
    if isinstance(decile, pd.Series) and len(decile) > 0:
        for g, r in decile.items():
            print(f"    {g}: {r:.4%}")

    print(f"\n{'─'*70}")
    print("  二、Top 10 因子重要性")
    print(f"{'─'*70}")
    for i, (_, row) in enumerate(imp.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature'][:20]:20s}  {row['importance']:8.0f}")

    print(f"\n{'─'*70}")
    print("  三、Top 5 推荐买入")
    print(f"{'─'*70}")
    for (code, name), row in top5.iterrows():
        print(f"  {code:>6s} {name:<8s}  预测收益: {row['pred_ret']:.2%}  收盘: ¥{row.get('close',0):.2f}")

    print(f"\n{'─'*70}")
    print("  四、Bottom 5 需警惕")
    print(f"{'─'*70}")
    for (code, name), row in bot5.iterrows():
        print(f"  {code:>6s} {name:<8s}  预测收益: {row['pred_ret']:.2%}  收盘: ¥{row.get('close',0):.2f}")

    print(f"\n{'─'*70}")
    print("  五、市场情绪")
    print(f"{'─'*70}")
    bullish = (latest['pred_ret'] > 0).mean()
    print(f"  看涨: {bullish:.1%} ({latest['pred_ret'].gt(0).sum()}/{len(latest)})")
    print(f"  看跌: {1-bullish:.1%} ({latest['pred_ret'].lt(0).sum()}/{len(latest)})")
    print(f"  平均预测收益: {latest['pred_ret'].mean():.2%}")
    print(f"  中位数: {latest['pred_ret'].median():.2%}")

    print(f"\n{'─'*70}")
    print("  免责声明: 本报告由 AI 模型自动生成，仅供参考，不构成投资建议")
    print(f"{'─'*70}\n")

    # 保存
    outdir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(outdir, exist_ok=True)
    latest.to_csv(os.path.join(outdir, "prediction.csv"))
    with open(os.path.join(outdir, "analysis_report.md"), "w") as f:
        f.write(f"""# Hyperion Quant — A 股市场分析报告

> 生成时间: {now}
> 数据区间: {data['date'].min().date()} ~ {data['date'].max().date()}

## 一、模型表现

| 指标 | 值 |
|------|----|
| 模型 | LightGBM |
| Rank IC | {rank_ic:.4f} |
| 多空收益差 | {spread:.2%} |

## 二、Top 5 推荐

| 代码 | 名称 | 预测收益 |
|-----|------|---------|
""")
        for (code, name), row in top5.iterrows():
            f.write(f"| {code} | {name} | {row['pred_ret']:.2%} |\n")
        f.write("\n## 三、Bottom 5\n\n| 代码 | 名称 | 预测收益 |\n|-----|------|---------|\n")
        for (code, name), row in bot5.iterrows():
            f.write(f"| {code} | {name} | {row['pred_ret']:.2%} |\n")

    print(f"  报告: {outdir}/analysis_report.md")
    return latest


if __name__ == "__main__":
    start = time.time()
    data = fetch_data_yahoo()
    if data is None:
        sys.exit(1)
    
    df = compute_features(data)
    
    model, Xtr, Xte, ytr, yte, preds, imp, df_model, feat_cols = train_lightgbm(df)
    
    rank_ic, decile, spread = evaluate(yte, preds)
    
    latest = predict_and_report(df_model, model, feat_cols, data, rank_ic, decile, spread)
    
    print(f"  总耗时: {time.time()-start:.1f} 秒")