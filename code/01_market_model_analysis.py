###############################################################################
#  東日本大震災前後の構造変化 ― 東京電力・関西電力のマーケットモデル (Python版)
#  code/01_market_model_analysis.R と同じ分析を Python で再現する
#  データ: data/SCmonthly.csv (2003年9月〜2020年10月, 月次)
#  収益率: 2003年10月〜2020年10月 (n = 205)
#
#  実行方法: プロジェクトのルートディレクトリを作業ディレクトリにして実行する
#    python code/01_market_model_analysis.py          # 図は output/ に保存
#    python code/01_market_model_analysis.py --show   # 図を画面にも表示
#  仕様と R との違い(P-1〜P-7): docs/python_port_requirements.md
###############################################################################

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

DATA = Path("data/SCmonthly.csv")
OUTPUT = Path("output")
QUAKE_LABEL = "11-Mar"                   # 東日本大震災(2011年3月11日)を含む月
FIRMS = {"TKYD": "東京電力", "KNSD": "関西電力"}
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


## ---- 0. データの読み込み -------------------------------------------------
def load_data(path=DATA):
    xy = pd.read_csv(path)
    assert len(xy) == 206, f"行数が想定(206)と異なる: {len(xy)}"
    assert xy.isna().sum().sum() == 0, "欠損値がある"
    return xy


## ---- 1. 価格から収益率へ (対数差分) --------------------------------------
## ---- 2. ダミー変数の作成 --------------------------------------------------
@dataclass
class Returns:
    x: np.ndarray      # 市場(TOPIX)の収益率
    y1: np.ndarray     # 東京電力の収益率
    y2: np.ndarray     # 関西電力の収益率
    lab: np.ndarray    # 収益率の対応月 ("3-Oct" 〜 "20-Oct")
    n: int             # 205
    q: int             # 震災月の位置. 0始まりなので 89 (R では 90)
    d: np.ndarray      # 定数項シフト用ダミー
    dx: np.ndarray     # 傾き(ベータ)シフト用の交差項 d*x
    tt: np.ndarray     # 株価の月を年の小数で表したもの(図の横軸)
    tr: np.ndarray     # 収益率の月を年の小数で表したもの

    def firms(self):
        return {"TKYD": self.y1, "KNSD": self.y2}


def decimal_year(ym):
    ## "3-Sep" は年が1桁で %y に合わず pd.to_datetime が失敗するため, 分割して変換する (P-1)
    parts = ym.str.split("-", expand=True)
    return (2000 + parts[0].astype(int) + (parts[1].map(MONTHS) - 1) / 12).to_numpy()


def make_returns(xy):
    x = np.diff(np.log(xy["TOPIX"].to_numpy()))
    y1 = np.diff(np.log(xy["TKYD"].to_numpy()))
    y2 = np.diff(np.log(xy["KNSD"].to_numpy()))
    n = len(x)
    lab = xy["ym"].to_numpy()[1:]
    ## 位置を直接書かずラベルで探す. インデックスが R と1ずれても結果は同じになる (P-2)
    q = int(np.flatnonzero(lab == QUAKE_LABEL)[0])
    d = (np.arange(n) >= q).astype(float)
    tt = decimal_year(xy["ym"])
    return Returns(x=x, y1=y1, y2=y2, lab=lab, n=n, q=q, d=d, dx=d * x, tt=tt, tr=tt[1:])


## ---- 回帰の共通部品 ----------------------------------------------------------
def ols(y, const=True, **cols):
    """lm() に相当. ols(y, d=d, x=x) は lm(y ~ d + x), const=False は `- 1` (定数項なし)"""
    X = pd.DataFrame(cols, index=range(len(y)))
    if const:
        X.insert(0, "(Intercept)", 1.0)
    return sm.OLS(np.asarray(y), X).fit()


def adj_r2(m, y):
    ## 分母を中心化TSSに統一した修正済み決定係数.
    ## statsmodels の rsquared_adj も定数項なしでは非中心化TSSを使い, 比較に使えない (P-3)
    k = len(m.params)
    return 1 - (m.ssr / (len(y) - k)) / (np.sum((y - y.mean()) ** 2) / (len(y) - 1))


def r_aic(m):
    ## R の AIC() と同じ値. statsmodels の aic は誤差分散をパラメータに数えず 2 小さい (P-4)
    return m.aic + 2


def coef_table(m):
    return pd.DataFrame({"Estimate": m.params, "Std. Error": m.bse,
                         "t value": m.tvalues, "Pr(>|t|)": m.pvalues})


## ---- 3. 時系列プロットの比較 ----------------------------------------------
def plot_figures(xy, r, outdir=OUTPUT):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False
    outdir.mkdir(exist_ok=True)
    quake = r.tr[r.q]                     # 震災の時点

    ## 左列: 株価水準, 右列: 対数収益率 (R の par(mfcol = c(3, 2)) と同じ並び)
    fig, ax = plt.subplots(3, 2, figsize=(11, 9))
    prices = [("TKYD", "東京電力 株価", "株価(円)"), ("KNSD", "関西電力 株価", "株価(円)"),
              ("TOPIX", "TOPIX", "指数")]
    for a, (col, title, ylab) in zip(ax[:, 0], prices):
        a.plot(r.tt, xy[col], color="black", lw=1)
        a.axvline(quake, color="red", ls="--")
        a.set(title=title, ylabel=ylab)
    rets = [(r.y1, "東京電力 収益率"), (r.y2, "関西電力 収益率"), (r.x, "TOPIX 収益率")]
    for a, (v, title) in zip(ax[:, 1], rets):
        a.plot(r.tr, v, color="black", lw=1)
        a.axvline(quake, color="red", ls="--")
        a.axhline(0, color="gray", lw=0.8)
        a.set(title=title, ylabel="対数収益率")
    for a in ax.flat:
        a.set_xticks(range(2005, 2021, 5))    # 年を整数で表示 (既定では 2007.5 のような小数になる)
    fig.tight_layout()
    fig.savefig(outdir / "fig_timeseries.png", dpi=150)

    ## 散布図(期間別の回帰直線つき)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    pre, post = r.d == 0, r.d == 1
    grid = np.array([r.x.min(), r.x.max()])
    for a, (code, y) in zip(ax, r.firms().items()):
        nm = FIRMS[code]
        a.scatter(r.x[pre], y[pre], facecolors="none", edgecolors="steelblue", label="震災前")
        a.scatter(r.x[post], y[post], color="firebrick", s=14, label="震災後")
        for sel, color in [(pre, "steelblue"), (post, "firebrick")]:
            b0, b1 = ols(y[sel], x=r.x[sel]).params
            a.plot(grid, b0 + b1 * grid, color=color, lw=2)
        a.set(title=nm, xlabel="TOPIX 収益率", ylabel=f"{nm} 収益率")
        a.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_scatter.png", dpi=150)
    return [outdir / "fig_timeseries.png", outdir / "fig_scatter.png"]


## ---- 4. 記述統計 -----------------------------------------------------------
def describe(r):
    rows = {}
    for name, v in [("TOPIX", r.x), ("TKYD", r.y1), ("KNSD", r.y2)]:
        for suffix, sel in [("", np.full(r.n, True)), ("_pre", r.d == 0), ("_post", r.d == 1)]:
            s = v[sel]
            rows[name + suffix] = {"n": len(s), "mean": s.mean(), "sd": s.std(ddof=1),
                                   "min": s.min(), "max": s.max()}
    return pd.DataFrame(rows).T


## ---- 5. モデル選択 (修正済み決定係数) --------------------------------------
MODELS = [                       # (ラベル, 説明変数, 定数項)
    ("a",            [],                True),
    ("a+bx",         ["x"],             True),
    ("a+cd+bx",      ["d", "x"],        True),
    ("a+bx+g dx",    ["x", "dx"],       True),
    ("a+cd+bx+g dx", ["d", "x", "dx"],  True),
    ("a+cd+g dx",    ["d", "dx"],       True),
    ("bx+g dx",      ["x", "dx"],       False),
    ("cd+bx+g dx",   ["d", "x", "dx"],  False),
    ("cd+g dx",      ["d", "dx"],       False),
    ("g dx",         ["dx"],            False),
]


def compare_models(y, r):
    rows = {}
    for label, terms, const in MODELS:
        m = ols(y, const=const, **{t: getattr(r, t) for t in terms})
        rows[label] = {"k": len(m.params), "adjR2": adj_r2(m, y),
                       "s": np.sqrt(m.scale), "AIC": r_aic(m)}
    table = pd.DataFrame(rows).T.astype({"k": int})
    return table.sort_values("adjR2", ascending=False, kind="stable")


## ---- 6. 主要モデルの推定 ---------------------------------------------------
def main_models(y, r):
    return {"base": ols(y, x=r.x),                          # 構造変化なし
            "full": ols(y, d=r.d, x=r.x, dx=r.dx),          # 完全モデル
            "best": ols(y, const=False, d=r.d, dx=r.dx)}    # 修正済みR^2 最大


## ---- 7. 構造変化の検定 (Chow検定 = c と g の同時検定) ----------------------
def chow_test(base, full):
    return anova_lm(base, full)


def subperiod_models(y, r):
    pre, post = r.d == 0, r.d == 1
    return {"pre": ols(y[pre], x=r.x[pre]), "post": ols(y[post], x=r.x[post])}


## ---- 8. 頑健性チェック -----------------------------------------------------
## (a) HC1(不均一分散のみ頑健)とHAC(不均一分散+系列相関に頑健)の標準誤差を OLS と並べる
def robust_se(m, lag=4):
    hc = m.get_robustcov_results(cov_type="HC1")
    ## use_correction=True で R の自前実装と同じ小標本補正 N/(N-k) になる (P-5)
    nw = m.get_robustcov_results(cov_type="HAC", maxlags=lag, use_correction=True)
    b = m.params.to_numpy()
    se_hc, se_nw = np.asarray(hc.bse), np.asarray(nw.bse)
    return pd.DataFrame({"coef": b, "se_OLS": m.bse.to_numpy(),
                         "se_HC1": se_hc, "t_HC1": b / se_hc,
                         "se_HAC": se_nw, "t_HAC": b / se_nw,
                         "p_HAC": 2 * stats.norm.sf(np.abs(b / se_nw))}, index=m.params.index)


## (b) 分散の変化: R の var.test(a, b) に相当 (SciPy に同名の関数はない, P-7)
def var_test(a, b):
    F = np.var(a, ddof=1) / np.var(b, ddof=1)
    df1, df2 = len(a) - 1, len(b) - 1
    p = 2 * min(stats.f.cdf(F, df1, df2), stats.f.sf(F, df1, df2))
    return {"F": F, "num_df": df1, "denom_df": df2, "p": p}


## (c)(d) 指定した月を除いて完全モデルを推定し直す
def exclusion_model(y, r, drop):
    keep = np.full(r.n, True)
    keep[drop] = False
    return ols(y[keep], d=r.d[keep], x=r.x[keep], dx=r.dx[keep])


## (e) 断点の位置をずらしたときの修正済み決定係数
def breakpoint_shift(y, r, shifts=range(-2, 3)):
    out = {}
    for s in shifts:
        qq = r.q + s
        dd = (np.arange(r.n) >= qq).astype(float)
        out[r.lab[qq]] = adj_r2(ols(y, dd=dd, x=r.x, ddx=dd * r.x), y)
    return pd.Series(out)


## (f) 残差の診断と, 震災前後のベータの差の検定
def diagnostics(y, r, lag=4):
    m = ols(y, d=r.d, x=r.x, dx=r.dx)
    bp_lm, bp_p, _, _ = het_breuschpagan(m.resid, m.model.exog)   # Koenker 版 (n R^2)
    bg_lm, bg_p, _, _ = acorr_breusch_godfrey(m, nlags=lag, result_object=False)
    ## 期間ごとに分散が違ってよい Welch 型の z 検定
    sub = subperiod_models(y, r)
    diff = sub["post"].params["x"] - sub["pre"].params["x"]
    z = diff / np.sqrt(sub["pre"].bse["x"] ** 2 + sub["post"].bse["x"] ** 2)
    return {"DW": durbin_watson(m.resid), "BP_LM": bp_lm, "BP_p": bp_p,
            "BG_LM": bg_lm, "BG_p": bg_p,
            "beta_diff": diff, "beta_z": z, "beta_p": 2 * stats.norm.sf(abs(z))}


## ---- 9. 補足: AIC によるステップワイズ ------------------------------------
def extract_aic(m):
    ## stepAIC が使う R の extractAIC() と同じ基準. statsmodels の aic とは定数が違う (P-4, P-6)
    return m.nobs * np.log(m.ssr / m.nobs) + 2 * len(m.params)


def step_forward(y, r, scope=("x", "d", "dx")):
    """MASS::stepAIC(lm(y ~ 1), direction = "forward") に相当 (statsmodels に同等の関数がない)"""
    terms, steps = [], []
    current = ols(y)
    while True:
        rows = {"<none>": {"Df": np.nan, "Sum of Sq": np.nan,
                           "RSS": current.ssr, "AIC": extract_aic(current)}}
        for t in scope:
            if t in terms:
                continue
            m = ols(y, **{v: getattr(r, v) for v in terms + [t]})
            rows[f"+ {t}"] = {"Df": 1, "Sum of Sq": current.ssr - m.ssr,
                              "RSS": m.ssr, "AIC": extract_aic(m)}
        table = pd.DataFrame(rows).T.sort_values("AIC", kind="stable")
        table["Df"] = table["Df"].astype("Int64")
        steps.append({"terms": list(terms), "AIC": extract_aic(current), "table": table})
        best = table.index[0]
        if best == "<none>" or len(terms) == len(scope):
            return current, steps
        terms.append(best[2:])
        current = ols(y, **{v: getattr(r, v) for v in terms})


## ---- 実行 --------------------------------------------------------------------
def section(title):
    print(f"\n## ---- {title} " + "-" * max(0, 70 - len(title)))


def print_model(name, m, y):
    print(f"\n--- {name} ---")
    print(coef_table(m).round(5))
    f_p = m.f_pvalue if m.df_model > 0 else np.nan
    print(f"adjR2(中心化)={adj_r2(m, y):.4f}  s={np.sqrt(m.scale):.4f}  "
          f"F={m.fvalue:.3f} (p={f_p:.4g})  n={int(m.nobs)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="東日本大震災前後のマーケットモデルの構造変化 (Python版)")
    parser.add_argument("--show", action="store_true", help="図を output/ に保存したうえで画面にも表示する")
    args = parser.parse_args(argv)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    pd.set_option("display.width", 120)

    section("0. データの読み込み")
    xy = load_data()
    print(xy.head(6)); print(xy.tail(6))

    section("1-2. 収益率とダミー変数")
    r = make_returns(xy)
    print(f"n={r.n}  震災月={r.lab[r.q]} (位置 {r.q}, R では {r.q + 1})  "
          f"pre={int((r.d == 0).sum())}  post={int((r.d == 1).sum())}")

    section("3. 時系列プロットの比較")
    for path in plot_figures(xy, r):
        print(f"saved: {path}")

    section("4. 記述統計")
    print(describe(r).round(4))

    section("5. モデル選択 (修正済み決定係数)")
    for code, y in r.firms().items():
        print(f"\n[{code} {FIRMS[code]}]")
        print(compare_models(y, r).round(4))

    section("6. 主要モデルの推定")
    fits = {code: main_models(y, r) for code, y in r.firms().items()}
    labels = {"base": "構造変化なし y ~ x", "full": "完全モデル y ~ d + x + dx",
              "best": "縮約モデル y ~ d + dx - 1"}
    for code, y in r.firms().items():
        for key, label in labels.items():
            print_model(f"{code} {label}", fits[code][key], y)

    section("7. 構造変化の検定 (Chow検定)")
    for code, y in r.firms().items():
        print(f"\n[{code}]")
        print(chow_test(fits[code]["base"], fits[code]["full"]))
    print("\n期間別に分けた推定")
    for code, y in r.firms().items():
        for period, m in subperiod_models(y, r).items():
            print(f"\n[{code} {period}]")
            print(coef_table(m).round(5))

    section("8. 頑健性チェック")
    print("\n(a) 頑健標準誤差 (完全モデル)")
    for code in r.firms():
        print(f"\n[{code}]")
        print(robust_se(fits[code]["full"]).round(4))
    print("\n(b) 分散の変化 (震災後 / 震災前)")
    for name, v in [("TKYD", r.y1), ("KNSD", r.y2), ("TOPIX", r.x)]:
        t = var_test(v[r.d == 1], v[r.d == 0])
        print(f"{name:5}: F = {t['F']:.5g}, num df = {t['num_df']}, denom df = {t['denom_df']}, "
              f"p-value = {t['p']:.4g}")
    print("\n(c) 2011年3月(暴落月)を除外")
    for code, y in r.firms().items():
        print_model(f"{code} n={r.n - 1}", exclusion_model(y, r, [r.q]), y[np.arange(r.n) != r.q])
    print("\n(d) 2011年3〜12月を除外")
    drop = np.arange(r.q, r.q + 10)        # 3月〜12月の10件 (R の q:(q+9))
    keep = ~np.isin(np.arange(r.n), drop)
    for code, y in r.firms().items():
        print_model(f"{code} n={int(keep.sum())}", exclusion_model(y, r, drop), y[keep])
    print("\n(e) 断点の位置をずらしたときの修正済み決定係数")
    for code, y in r.firms().items():
        shifts = breakpoint_shift(y, r)
        print(f"{code} : " + " ".join(f"{k}={v:.4f}" for k, v in shifts.items()))
    print("\n(f) 残差の診断と, 期間別ベータの差の検定 (完全モデル)")
    print(pd.DataFrame({code: diagnostics(y, r) for code, y in r.firms().items()}).T.round(4))

    section("9. 補足: AIC によるステップワイズ")
    for code, y in r.firms().items():
        final, steps = step_forward(y, r)
        for i, s in enumerate(steps):
            rhs = " + ".join(s["terms"]) or "1"
            print(f"\n{'Start' if i == 0 else 'Step'}:  AIC={s['AIC']:.2f}\n{code} ~ {rhs}\n")
            print(s["table"].round(4))
        print(f"\n最終モデルの係数:\n{final.params.round(6)}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
