"""R 版 (code/01_market_model_analysis.R) の出力と数値が一致することを確かめる.

参照値は docs/python_port_requirements.md §8 (R を実行して得た値).
許容誤差: 係数・SE・R²・σ は 1e-4, F値・AIC・検定統計量は 1e-2, p値は相対 1%.
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analysis", ROOT / "code" / "01_market_model_analysis.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)

COEF = dict(abs=1e-4)
STAT = dict(abs=1e-2)
PVAL = dict(rel=0.01)


@pytest.fixture(scope="module")
def xy():
    return analysis.load_data(ROOT / "data" / "SCmonthly.csv")


@pytest.fixture(scope="module")
def r(xy):
    return analysis.make_returns(xy)


## ---- 8.1 データ ---------------------------------------------------------------
def test_sample_split(r):
    assert r.n == 205
    assert r.lab[r.q] == "11-Mar"
    assert r.q + 1 == 90                       # R は1始まり
    assert (r.d == 0).sum() == 89 and (r.d == 1).sum() == 116
    assert r.lab[0] == "3-Oct" and r.lab[-1] == "20-Oct"


def test_decimal_year(xy, r):
    assert r.tt[0] == pytest.approx(2003 + 8 / 12)     # 2003年9月
    assert r.tr[r.q] == pytest.approx(2011 + 2 / 12)   # 2011年3月


@pytest.mark.parametrize("row, sd", [
    ("TOPIX_pre", 0.0541), ("TOPIX_post", 0.0480),
    ("TKYD_pre", 0.0537), ("TKYD_post", 0.1973),
    ("KNSD_pre", 0.0456), ("KNSD_post", 0.1017),
])
def test_describe_sd(r, row, sd):
    assert analysis.describe(r).loc[row, "sd"] == pytest.approx(sd, **COEF)


## ---- 8.2 モデル比較 -----------------------------------------------------------
COMPARE = {   # ラベル: (修正済みR², s, AIC), R の出力順
    "TKYD": [("cd+g dx", 0.1860, 0.1376, -227.4436), ("cd+bx+g dx", 0.1830, 0.1378, -225.7097),
             ("a+cd+g dx", 0.1820, 0.1379, -225.4526), ("a+cd+bx+g dx", 0.1790, 0.1382, -223.7174),
             ("a+bx+g dx", 0.1765, 0.1384, -224.0748), ("g dx", 0.1735, 0.1387, -225.3137),
             ("bx+g dx", 0.1705, 0.1389, -223.5745), ("a+cd+bx", 0.1035, 0.1444, -206.6693),
             ("a+bx", 0.1029, 0.1444, -207.5227), ("a", 0.0000, 0.1525, -186.2479)],
    "KNSD": [("cd+g dx", 0.1450, 0.0760, -470.9983), ("cd+bx+g dx", 0.1414, 0.0761, -469.1299),
             ("a+cd+g dx", 0.1409, 0.0762, -469.0255), ("g dx", 0.1392, 0.0762, -470.5864),
             ("a+cd+bx+g dx", 0.1372, 0.0763, -467.1588), ("a+bx+g dx", 0.1359, 0.0764, -467.8246),
             ("bx+g dx", 0.1355, 0.0764, -468.7164), ("a+bx", 0.0775, 0.0789, -455.4093),
             ("a+cd+bx", 0.0773, 0.0789, -454.3674), ("a", 0.0000, 0.0822, -439.8662)],
}


@pytest.mark.parametrize("code", ["TKYD", "KNSD"])
def test_compare_models(r, code):
    table = analysis.compare_models(r.firms()[code], r)
    assert list(table.index) == [row[0] for row in COMPARE[code]]
    for label, adj, s, aic in COMPARE[code]:
        assert table.loc[label, "adjR2"] == pytest.approx(adj, **COEF)
        assert table.loc[label, "s"] == pytest.approx(s, **COEF)
        assert table.loc[label, "AIC"] == pytest.approx(aic, **STAT)


def test_statsmodels_adj_r2_is_not_comparable(r):
    """P-3: 定数項なしモデルでは statsmodels の値が中心化TSSの値と異なる"""
    m = analysis.ols(r.y2, const=False, d=r.d, dx=r.dx)
    assert m.rsquared_adj == pytest.approx(0.1423, **COEF)
    assert analysis.adj_r2(m, r.y2) == pytest.approx(0.1450, **COEF)


## ---- 8.3 主要モデルの係数 -----------------------------------------------------
MAIN = {   # (モデル, 項): (TKYD 係数, SE, KNSD 係数, SE)
    ("base", "(Intercept)"): (-0.01286, 0.01010, -0.00443, 0.00552),
    ("base", "x"):           (0.98574, 0.19953, 0.46421, 0.10900),
    ("full", "(Intercept)"): (-0.00127, 0.01465, 0.00136, 0.00809),
    ("full", "d"):           (-0.02481, 0.01951, -0.01234, 0.01078),
    ("full", "x"):           (0.13879, 0.27231, 0.05438, 0.15038),
    ("full", "dx"):          (1.69139, 0.38232, 0.81879, 0.21114),
    ("best", "d"):           (-0.02608, 0.01283, -0.01098, 0.00709),
    ("best", "dx"):          (1.83019, 0.26722, 0.87317, 0.14753),
}


@pytest.mark.parametrize("key", list(MAIN))
def test_main_model_coefficients(r, key):
    model, term = key
    tk_b, tk_se, kn_b, kn_se = MAIN[key]
    for y, b, se in [(r.y1, tk_b, tk_se), (r.y2, kn_b, kn_se)]:
        m = analysis.main_models(y, r)[model]
        assert m.params[term] == pytest.approx(b, **COEF)
        assert m.bse[term] == pytest.approx(se, **COEF)


@pytest.mark.parametrize("code, base, full", [("TKYD", 0.1029, 0.1790), ("KNSD", 0.0775, 0.1372)])
def test_main_model_adj_r2(r, code, base, full):
    y = r.firms()[code]
    fits = analysis.main_models(y, r)
    assert analysis.adj_r2(fits["base"], y) == pytest.approx(base, **COEF)
    assert analysis.adj_r2(fits["full"], y) == pytest.approx(full, **COEF)


## ---- 8.4 検定と頑健性 ---------------------------------------------------------
@pytest.mark.parametrize("code, F, p", [("TKYD", 10.404, 5.016e-05), ("KNSD", 8.0254, 4.434e-04)])
def test_chow(r, code, F, p):
    fits = analysis.main_models(r.firms()[code], r)
    table = analysis.chow_test(fits["base"], fits["full"])
    assert table["df_resid"].tolist() == [203, 201]
    assert table["F"].iloc[1] == pytest.approx(F, **STAT)
    assert table["Pr(>F)"].iloc[1] == pytest.approx(p, **PVAL)


@pytest.mark.parametrize("code, pre, post", [("TKYD", 0.1388, 1.8302), ("KNSD", 0.0544, 0.8732)])
def test_subperiod_beta(r, code, pre, post):
    sub = analysis.subperiod_models(r.firms()[code], r)
    assert sub["pre"].params["x"] == pytest.approx(pre, **COEF)
    assert sub["post"].params["x"] == pytest.approx(post, **COEF)


ROBUST = {   # 定数項, d, x, dx の順
    "TKYD": {"se_HC1": [0.00562, 0.01891, 0.14829, 0.57459],
             "se_HAC": [0.00551, 0.02001, 0.08797, 0.56714]},
    "KNSD": {"se_HC1": [0.00483, 0.00993, 0.10921, 0.21663],
             "se_HAC": [0.00481, 0.00955, 0.06900, 0.18717]},
}


@pytest.mark.parametrize("code", ["TKYD", "KNSD"])
def test_robust_se(r, code):
    table = analysis.robust_se(analysis.main_models(r.firms()[code], r)["full"])
    for col, expected in ROBUST[code].items():
        assert table[col].to_numpy() == pytest.approx(expected, **COEF)


def test_robust_p_values(r):
    tk = analysis.robust_se(analysis.main_models(r.y1, r)["full"])
    kn = analysis.robust_se(analysis.main_models(r.y2, r)["full"])
    assert tk.loc["dx", "p_HAC"] == pytest.approx(0.00286, **PVAL)
    assert kn.loc["dx", "p_HAC"] < 5e-5


def test_hac_without_correction_differs(r):
    """P-5: use_correction を付けないと R と一致しない"""
    m = analysis.main_models(r.y1, r)["full"]
    se = np.asarray(m.get_robustcov_results(cov_type="HAC", maxlags=4).bse)
    assert abs(se[3] - 0.56714) > 1e-3


@pytest.mark.parametrize("name, F, p", [("TKYD", 13.497, None), ("KNSD", 4.9715, 1.599e-13),
                                        ("TOPIX", 0.78785, 0.2296)])
def test_var_test(r, name, F, p):
    v = {"TKYD": r.y1, "KNSD": r.y2, "TOPIX": r.x}[name]
    t = analysis.var_test(v[r.d == 1], v[r.d == 0])
    assert (t["num_df"], t["denom_df"]) == (115, 88)
    assert t["F"] == pytest.approx(F, **STAT)
    if p is None:
        assert t["p"] < 2.2e-16                 # R の表示は "< 2.2e-16" (P-7)
    else:
        assert t["p"] == pytest.approx(p, **PVAL)


@pytest.mark.parametrize("code, b, se", [("TKYD", 1.19935, 0.28012), ("KNSD", 0.78894, 0.21288)])
def test_exclude_march_2011(r, code, b, se):
    m = analysis.exclusion_model(r.firms()[code], r, [r.q])
    assert m.nobs == 204
    assert m.params["dx"] == pytest.approx(b, **COEF)
    assert m.bse["dx"] == pytest.approx(se, **COEF)


@pytest.mark.parametrize("code, b, se", [("TKYD", 1.17001, 0.24531), ("KNSD", 0.80852, 0.20803)])
def test_exclude_march_to_december_2011(r, code, b, se):
    m = analysis.exclusion_model(r.firms()[code], r, np.arange(r.q, r.q + 10))
    assert m.nobs == 195
    assert m.params["dx"] == pytest.approx(b, **COEF)
    assert m.bse["dx"] == pytest.approx(se, **COEF)


@pytest.mark.parametrize("code, expected", [
    ("TKYD", [0.1796, 0.1796, 0.1790, 0.1081, 0.1079]),
    ("KNSD", [0.1386, 0.1387, 0.1372, 0.1226, 0.1210]),
])
def test_breakpoint_shift(r, code, expected):
    shifts = analysis.breakpoint_shift(r.firms()[code], r)
    assert list(shifts.index) == ["11-Jan", "11-Feb", "11-Mar", "11-Apr", "11-May"]
    assert shifts.to_numpy() == pytest.approx(expected, **COEF)


@pytest.mark.parametrize("code, dw, bp, bp_p, bg, bg_p, diff, z", [
    ("TKYD", 2.006, 8.75, 0.0328, 2.07, 0.7225, 1.6914, 4.694),
    ("KNSD", 2.002, 8.88, 0.0310, 1.32, 0.8580, 0.8188, 4.052),
])
def test_diagnostics(r, code, dw, bp, bp_p, bg, bg_p, diff, z):
    res = analysis.diagnostics(r.firms()[code], r)
    assert res["DW"] == pytest.approx(dw, abs=1e-3)
    assert res["BP_LM"] == pytest.approx(bp, **STAT)
    assert res["BP_p"] == pytest.approx(bp_p, **PVAL)
    assert res["BG_LM"] == pytest.approx(bg, **STAT)
    assert res["BG_p"] == pytest.approx(bg_p, **PVAL)
    assert res["beta_diff"] == pytest.approx(diff, **COEF)
    assert res["beta_z"] == pytest.approx(z, abs=1e-3)


## ---- 8.5 ステップワイズ -------------------------------------------------------
@pytest.mark.parametrize("code, start, step1, const, dx", [
    ("TKYD", -770.01, -809.59, -0.0153, 1.8090),
    ("KNSD", -1023.63, -1053.47, -0.005614, 0.862596),
])
def test_step_forward(r, code, start, step1, const, dx):
    final, steps = analysis.step_forward(r.firms()[code], r)
    assert len(steps) == 2
    assert steps[0]["AIC"] == pytest.approx(start, **STAT)
    assert list(steps[0]["table"].index) == ["+ dx", "+ x", "<none>", "+ d"]
    assert steps[1]["terms"] == ["dx"]
    assert steps[1]["AIC"] == pytest.approx(step1, **STAT)
    assert steps[1]["table"].index[0] == "<none>"
    assert final.params["(Intercept)"] == pytest.approx(const, **COEF)
    assert final.params["dx"] == pytest.approx(dx, **COEF)


def test_step_forward_first_table_tkyd(r):
    _, steps = analysis.step_forward(r.y1, r)
    t = steps[0]["table"]
    assert t["RSS"].to_numpy() == pytest.approx([3.8739, 4.2357, 4.7450, 4.7315], **COEF)
    assert t["AIC"].to_numpy() == pytest.approx([-809.59, -791.29, -770.01, -768.60], **STAT)


## ---- 8.6 図 -------------------------------------------------------------------
def test_plot_figures(xy, r, tmp_path):
    paths = analysis.plot_figures(xy, r, outdir=tmp_path)
    assert [p.name for p in paths] == ["fig_timeseries.png", "fig_scatter.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
