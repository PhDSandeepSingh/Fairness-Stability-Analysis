import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Paper 1 - How much of a fairness-mitigation gain survives split noise?
**A fairness-stability audit protocol on four UCI tabular benchmarks** (ICECRF-2027, Track 1, topic: *Ethical AI, algorithmic bias, fairness and transparency*).

**Protocol in one paragraph.** For every dataset we draw R = 20 repeated stratified splits (fit / post-processing / test). On each split we train three base learners and apply four mitigation families. Every mitigated model is compared with the *same-learner baseline on the same split*, giving a paired difference per split. We then report the mean paired difference, a Nadeau-Bengio corrected confidence interval and test, the single-split win rate, and the single-split sign-flip rate (how often one split would have led to the opposite conclusion).

Design decisions fixed *before* any result was inspected are listed in the next cell.
''')

md(r'''
## 0. Fixed design decisions (pre-specified)
| Item | Decision |
|---|---|
| Sensitive attribute | Adult: sex (Male = 1). German Credit: age >= 25 (=1). Bank Marketing: age in [25, 60] (=1) versus outside. Credit Default: sex (Male = 1). |
| Sensitive attribute as a feature | Kept in the feature set for all models. |
| Favourable outcome (y = 1) | Adult: income > 50K. German: good credit. Bank: term-deposit subscribed. Credit: *no* default next month. |
| Bank Marketing `duration` | Dropped (UCI documentation says it is unknown before the call, so it leaks the outcome). |
| Duplicates | Exact duplicate rows dropped before splitting (counts reported). |
| Splits | R repeated stratified splits on (group x label): 30 % test; remaining 70 % divided 75 % fit / 25 % post-processing calibration. |
| Learners (fixed hyper-parameters, no tuning) | Logistic regression, random forest, histogram gradient boosting. |
| Mitigations | Reweighing (Kamiran-Calders), ThresholdOptimizer (demographic parity, equalized odds), ExponentiatedGradient (demographic parity, equalized odds; logistic regression base only). |
| Fairness metrics | Demographic parity difference (DPD), equalized odds difference (EOD). Utility: balanced accuracy (BA). Lower DPD/EOD is better. |
''')

code(r'''
import os, sys, io, json, zipfile, hashlib, platform, warnings, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
if "google.colab" in sys.modules:
    %pip -q install fairlearn xlrd
import sklearn, scipy, fairlearn, matplotlib
print({k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy, fairlearn=fairlearn, matplotlib=matplotlib).items()})
print(platform.platform(), "| Python", sys.version.split()[0])
''')

code(r'''
SEED = 2027
N_REPEATS = int(os.environ.get("N_REPEATS", 20))
TEST_FRAC, POST_FRAC = 0.30, 0.25          # post = share of the non-test part
DATASETS = os.environ.get("DATASETS", "adult,german,bank,credit").split(",")
N_JOBS = int(os.environ.get("N_JOBS", 4))
MAX_SEC = float(os.environ.get("MAX_SECONDS", 1e9))   # time budget per invocation (the run is resumable)
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT = Path("results"); OUT.mkdir(exist_ok=True)
UCI = {"adult": (2, "adult"), "german": (144, "statlog+german+credit+data"),
       "bank": (222, "bank+marketing"), "credit": (350, "default+of+credit+card+clients")}
NICE = {"adult": "Adult", "german": "German Credit", "bank": "Bank Marketing", "credit": "Credit Default"}
SENS = {"adult": "sex (male)", "german": "age >= 25", "bank": "age 25-60", "credit": "sex (male)"}

def ensure_data(key):
    pid, name = UCI[key]
    d = DATA_DIR / name
    if not (d.exists() and any(d.iterdir())):
        d.mkdir(parents=True, exist_ok=True)
        url = f"https://archive.ics.uci.edu/static/public/{pid}/{name}.zip"
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())).extractall(d)
    for z in list(d.rglob("*.zip")):                     # bank.zip contains nested zips
        tgt = z.parent
        if not any(p for p in tgt.iterdir() if p.suffix in (".csv", ".data")) or z.name.startswith("bank-additional"):
            zipfile.ZipFile(z).extractall(tgt)
    return d

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
''')

code(r'''
def load_adult(d):
    cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
            "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
    a = pd.read_csv(d/"adult.data", header=None, names=cols, skipinitialspace=True)
    t = pd.read_csv(d/"adult.test", header=None, names=cols, skipinitialspace=True, skiprows=1)
    df = pd.concat([a, t], ignore_index=True)
    df["income"] = df["income"].str.rstrip(".")
    n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
    y = (df.pop("income") == ">50K").astype(int).values
    df = df.drop(columns=["fnlwgt", "education"]).replace("?", "Unknown")
    s = (df["sex"] == "Male").astype(int).values
    return df, y, s, n0 - len(df), [d/"adult.data", d/"adult.test"]

def load_german(d):
    cols = ["status","duration","credit_history","purpose","credit_amount","savings","employment","installment_rate",
            "personal_status_sex","other_debtors","residence_since","property","age","other_installment_plans",
            "housing","existing_credits","job","num_dependents","telephone","foreign_worker","cls"]
    df = pd.read_csv(d/"german.data", sep=r"\s+", header=None, names=cols)
    n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
    y = (df.pop("cls") == 1).astype(int).values
    s = (df["age"] >= 25).astype(int).values
    return df, y, s, n0 - len(df), [d/"german.data"]

def load_bank(d):
    p = next(d.rglob("bank-additional-full.csv"))
    df = pd.read_csv(p, sep=";")
    n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
    y = (df.pop("y") == "yes").astype(int).values
    df = df.drop(columns=["duration"])
    s = df["age"].between(25, 60).astype(int).values
    return df, y, s, n0 - len(df), [p]

def load_credit(d):
    p = next(d.glob("*.xls*"))
    df = pd.read_excel(p, header=1).drop(columns="ID")
    n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
    tgt = "default payment next month"
    y = (df.pop(tgt) == 0).astype(int).values            # favourable = no default
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    for c in ["SEX", "EDUCATION", "MARRIAGE"]: df[c] = df[c].astype(str)
    s = (df["SEX"] == "1").astype(int).values
    return df, y, s, n0 - len(df), [p]

LOADERS = {"adult": load_adult, "german": load_german, "bank": load_bank, "credit": load_credit}

summary, hashes = [], {}
for k in DATASETS:
    d = ensure_data(k); X, y, s, ndup, files = LOADERS[k](d)
    hashes[k] = {f.name: sha256(f) for f in files}
    summary.append(dict(dataset=NICE[k], n=len(X), n_features=X.shape[1], dup_removed=ndup, sensitive=SENS[k],
                        group1_share=s.mean(), fav_rate=y.mean(), fav_rate_g1=y[s==1].mean(), fav_rate_g0=y[s==0].mean(),
                        n_g0=int((s==0).sum())))
DS = pd.DataFrame(summary); DS.to_csv(OUT/"table_datasets.csv", index=False)
json.dump(hashes, open(OUT/"data_sha256.json", "w"), indent=2)
DS.round(3)
''')

code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from joblib import Parallel, delayed

def make_prep(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([("n", StandardScaler(), num),
                              ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), cat)])

def learners(seed):
    return {"LR": LogisticRegression(max_iter=2000, C=1.0),
            "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=seed),
            "HGB": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)}

def reweigh(y, s):
    w = np.ones(len(y))
    for g in (0, 1):
        for c in (0, 1):
            m = (s == g) & (y == c)
            if m.any(): w[m] = ((s == g).mean() * (y == c).mean()) / m.mean()
    return w

def metrics(y, yh, s):
    return dict(acc=accuracy_score(y, yh), ba=balanced_accuracy_score(y, yh),
                dpd=demographic_parity_difference(y, yh, sensitive_features=s),
                eod=equalized_odds_difference(y, yh, sensitive_features=s))

def one_split(key, r):
    d = ensure_data(key); X, y, s, *_ = LOADERS[key](d)
    seed = SEED + 1000 * r
    strat = y * 2 + s
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=TEST_FRAC, stratify=strat, random_state=seed)
    i_fit, i_po = train_test_split(i_tr, test_size=POST_FRAC, stratify=strat[i_tr], random_state=seed + 1)
    prep = make_prep(X).fit(X.iloc[i_fit])
    Xf, Xp, Xt = (prep.transform(X.iloc[i]) for i in (i_fit, i_po, i_te))
    yf, yp, yt = y[i_fit], y[i_po], y[i_te]; sf, sp, st = s[i_fit], s[i_po], s[i_te]
    rows = []
    def add(model, method, yh, secs):
        rows.append(dict(dataset=key, rep=r, model=model, method=method, secs=secs, **metrics(yt, yh, st)))
    for name, mdl in learners(seed).items():
        t0 = time.time(); mdl.fit(Xf, yf); add(name, "Baseline", mdl.predict(Xt), time.time() - t0)
        # post-hoc sensitivity reference: same model, single (group-blind) threshold tuned for balanced accuracy on the post split
        pp = mdl.predict_proba(Xp)[:, 1]; grid = np.unique(np.quantile(pp, np.linspace(0.01, 0.99, 99)))
        thr = grid[np.argmax([balanced_accuracy_score(yp, (pp >= t).astype(int)) for t in grid])]
        add(name, "Baseline-Thr", (mdl.predict_proba(Xt)[:, 1] >= thr).astype(int), 0.0)
        t0 = time.time(); m2 = learners(seed)[name]; m2.fit(Xf, yf, sample_weight=reweigh(yf, sf))
        add(name, "Reweighing", m2.predict(Xt), time.time() - t0)
        for cons, lab in (("demographic_parity", "ThreshOpt-DP"), ("equalized_odds", "ThreshOpt-EO")):
            t0 = time.time()
            to = ThresholdOptimizer(estimator=mdl, constraints=cons, objective="balanced_accuracy_score",
                                    prefit=True, predict_method="predict_proba")
            to.fit(Xp, yp, sensitive_features=sp)
            add(name, lab, to.predict(Xt, sensitive_features=st, random_state=seed), time.time() - t0)
    for cons, lab in ((DemographicParity(), "ExpGrad-DP"), (EqualizedOdds(), "ExpGrad-EO")):
        t0 = time.time()
        eg = ExponentiatedGradient(LogisticRegression(max_iter=2000, C=1.0), constraints=cons, eps=0.01)
        eg.fit(Xf, yf, sensitive_features=sf)
        add("LR", lab, eg.predict(Xt, random_state=seed), time.time() - t0)
    return rows
''')

code(r'''
# quick smoke test on two splits of the smallest dataset before the full run
t0 = time.time(); test_rows = one_split(DATASETS[-1] if "german" not in DATASETS else "german", 0)
print(f"one split: {time.time()-t0:.1f}s, rows: {len(test_rows)}")
pd.DataFrame(test_rows).round(3).head(12)
''')

code(r'''
RAW = OUT / "raw_runs.csv"
done = set()
if RAW.exists():
    prev = pd.read_csv(RAW); done = set(zip(prev.dataset, prev.rep)); rows_all = prev.to_dict("records")
else:
    rows_all = []
todo = [(k, r) for k in DATASETS for r in range(N_REPEATS) if (k, r) not in done]
print(len(todo), "splits to run")
t0 = time.time()
CH = max(N_JOBS, 1)
for i in range(0, len(todo), CH):
    if time.time() - t0 > MAX_SEC: print("time budget reached - re-run to resume"); break
    chunk = todo[i:i+CH]
    for res in Parallel(n_jobs=N_JOBS)(delayed(one_split)(k, r) for k, r in chunk): rows_all.extend(res)
    pd.DataFrame(rows_all).to_csv(RAW, index=False)
    print(f"{min(i+CH, len(todo))}/{len(todo)} done, {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows_all); R.shape
''')

md(r'''
## Analysis
For each (dataset, learner, mitigation) we compute the paired difference Delta = metric(mitigated) - metric(same-learner baseline) per split, so negative Delta for DPD/EOD means *fairer*. Inference uses the Nadeau-Bengio corrected resampled t-test (variance inflated by 1/R + n_test/n_train) because repeated splits are not independent. P-values are Holm-adjusted within each metric across all comparisons. The sign-flip rate is the share of single splits whose Delta has the opposite sign to the mean Delta.
''')

code(r'''
from scipy import stats
NTE_NTR = TEST_FRAC / ((1 - TEST_FRAC) * (1 - POST_FRAC))   # test size / fit-split size (0.30 / 0.525)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, idx in enumerate(o):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj

base = R[R.method == "Baseline"].set_index(["dataset", "rep", "model"])
rows = []
for (ds, mdl, meth), g in R[~R.method.isin(["Baseline", "Baseline-Thr"])].groupby(["dataset", "model", "method"]):
    g = g.set_index("rep"); b = base.xs((ds, mdl), level=("dataset", "model")).loc[g.index]
    for met in ("dpd", "eod", "ba"):
        d = (g[met] - b[met]).values; n = len(d); m = d.mean(); sd = d.std(ddof=1)
        se = np.sqrt((1/n + NTE_NTR) * sd**2) if sd > 0 else np.nan
        t = m / se if se and se > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0
        h = stats.t.ppf(0.975, n - 1) * se if np.isfinite(se) else np.nan
        try: pw = stats.wilcoxon(d).pvalue
        except Exception: pw = 1.0
        pn = stats.ttest_1samp(d, 0).pvalue if sd > 0 else 1.0     # naive paired t-test, ignores split overlap
        better = (d < 0) if met != "ba" else (d > 0)
        win = better.mean(); flip = ((np.sign(d) != np.sign(m)) & (d != 0)).mean()
        rows.append(dict(dataset=ds, model=mdl, method=meth, metric=met, n=n, mean_delta=m, sd_delta=sd,
                         ci_lo=m - h, ci_hi=m + h, p_nb=p, p_wilcoxon=pw, p_naive=pn, win_rate=win, sign_flip=flip,
                         base_mean=b[met].mean(), base_sd=b[met].std(ddof=1)))
T = pd.DataFrame(rows)
for met in ("dpd", "eod", "ba"):
    mk = T.metric == met; T.loc[mk, "p_holm"] = holm(T.loc[mk, "p_nb"].values)
    T.loc[mk, "p_naive_holm"] = holm(T.loc[mk, "p_naive"].values); T.loc[mk, "p_wilcoxon_holm"] = holm(T.loc[mk, "p_wilcoxon"].values)
def verdict(r):
    if r.metric == "ba": return ""
    if r.p_holm < 0.05: return "robust gain" if r.mean_delta < 0 else "robust harm"
    return "indistinguishable from noise"
T["verdict"] = T.apply(verdict, axis=1)
T.to_csv(OUT / "table_paired_deltas.csv", index=False)
T[T.metric.isin(["dpd","eod"])].groupby(["metric","verdict"]).size().unstack(fill_value=0)
''')

code(r'''
# Post-hoc sensitivity analysis (added after the primary analysis): ThresholdOptimizer versus a threshold-tuned,
# group-blind baseline that uses the same post-processing split and the same balanced-accuracy objective.
bt = R[R.method == "Baseline-Thr"].set_index(["dataset", "rep", "model"])
srows = []
for (ds, mdl, meth), g in R[R.method.str.startswith("ThreshOpt")].groupby(["dataset", "model", "method"]):
    g = g.set_index("rep"); b = bt.xs((ds, mdl), level=("dataset", "model")).loc[g.index]
    for met in ("ba", "dpd", "eod"):
        d = (g[met] - b[met]).values; n = len(d); m = d.mean(); sd = d.std(ddof=1)
        se = np.sqrt((1/n + NTE_NTR) * sd**2); t = m / se if se > 0 else np.nan
        srows.append(dict(dataset=ds, model=mdl, method=meth, metric=met, mean_delta=m,
                          p_nb=2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0))
SENSI = pd.DataFrame(srows)
for met in ("ba", "dpd", "eod"):
    mk = SENSI.metric == met; SENSI.loc[mk, "p_holm"] = holm(SENSI.loc[mk, "p_nb"].values)
SENSI.to_csv(OUT / "table_sensitivity_tuned_threshold.csv", index=False)
# compare balanced-accuracy change against the 0.5-threshold baseline vs the tuned-threshold baseline
ba05 = T[(T.metric == "ba") & T.method.str.startswith("ThreshOpt")].mean_delta
print("ThreshOpt BA change vs 0.5-threshold baseline: mean %.3f; vs tuned-threshold baseline: mean %.3f"
      % (ba05.mean(), SENSI[SENSI.metric == "ba"].mean_delta.mean()))
SENSI.pivot_table(index=["dataset", "method"], columns="metric", values="mean_delta").round(3)
''')

code(r'''
import matplotlib; matplotlib.use("Agg") if "google.colab" not in sys.modules else None
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9})
# Table: mean +/- SD of metrics per dataset/learner/method
S = R.groupby(["dataset","model","method"]).agg(ba=("ba","mean"), ba_sd=("ba","std"), dpd=("dpd","mean"), dpd_sd=("dpd","std"),
                                                 eod=("eod","mean"), eod_sd=("eod","std"), runs=("rep","nunique")).reset_index()
S.to_csv(OUT / "table_summary_metrics.csv", index=False)

# Figure 1: paired change in EOD and DPD per method (all learners pooled per dataset)
order = ["Reweighing","ThreshOpt-DP","ThreshOpt-EO","ExpGrad-DP","ExpGrad-EO"]
fig, axes = plt.subplots(2, len(DATASETS), figsize=(1.6*len(DATASETS), 4.4), sharey="row", squeeze=False)
bq = R[R.method == "Baseline"].set_index(["dataset","rep","model"])
for j, ds in enumerate(DATASETS):
    for i, met in enumerate(("dpd","eod")):
        ax = axes[i, j]; data = []
        for me in order:
            g = R[(R.dataset==ds)&(R.method==me)].set_index(["dataset","rep","model"])
            data.append((g[met] - bq.loc[g.index, met]).values)
        ax.boxplot(data, labels=[o.replace("ThreshOpt","TO").replace("ExpGrad","EG").replace("Reweighing","RW") for o in order], showfliers=False)
        ax.axhline(0, color="k", lw=0.7); ax.tick_params(axis="x", rotation=60, labelsize=7); ax.tick_params(axis="y", labelsize=7)
        if j == 0: ax.set_ylabel(r"$\Delta$" + met.upper() + " vs. baseline")
        if i == 0: ax.set_title(NICE[ds], fontsize=8)
fig.text(0.5, 0.005, "negative = fairer than the same-learner baseline on the same split", ha="center", fontsize=7)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT/"fig_paired_change.png"); plt.savefig(OUT/"fig_paired_change.pdf"); plt.show()

# Figure 2: single-split sign-flip rate
F = T[T.metric.isin(["dpd","eod"])]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for k, met in enumerate(("dpd","eod")):
    x = F[F.metric==met].groupby("dataset").sign_flip.apply(lambda v: v.values)
    ax.boxplot([x.get(d, np.array([])) for d in DATASETS], positions=np.arange(len(DATASETS)) + (k-0.5)*0.32, widths=0.28,
               patch_artist=True, boxprops=dict(facecolor=["#9ecae1","#fdae6b"][k]), medianprops=dict(color="k"), showfliers=True)
ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels([NICE[d] for d in DATASETS], fontsize=8); ax.set_ylabel("sign-flip rate", fontsize=8)
ax.legend([plt.Rectangle((0,0),1,1,fc="#9ecae1"), plt.Rectangle((0,0),1,1,fc="#fdae6b")], ["DPD","EOD"], frameon=False)
plt.tight_layout(); plt.savefig(OUT/"fig_sign_flip.png"); plt.savefig(OUT/"fig_sign_flip.pdf"); plt.show()
''')

code(r'''
# Headline numbers used in the paper
H = T[T.metric.isin(["dpd","eod"])]
print("comparisons per metric:", H.groupby("metric").size().to_dict())
print(H.groupby(["metric","verdict"]).size().unstack(fill_value=0))
print("significant after Holm - corrected NB:", (H.p_holm < .05).groupby(H.metric).sum().to_dict(),
      "| naive paired t:", (H.p_naive_holm < .05).groupby(H.metric).sum().to_dict(),
      "| Wilcoxon:", (H.p_wilcoxon_holm < .05).groupby(H.metric).sum().to_dict())
print("median sign-flip rate by metric:", H.groupby("metric").sign_flip.median().round(3).to_dict())
print("share of comparisons with sign-flip >= 0.20:", (H.sign_flip >= 0.20).groupby(H.metric).mean().round(3).to_dict())
# how often would a single split declare an improvement even though the corrected test finds none?
ns = H[H.verdict == "indistinguishable from noise"]
print("noise-level comparisons: mean single-split win rate =", ns.win_rate.mean().round(3))
json.dump({"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED, "n_repeats": N_REPEATS,
           "packages": {k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy, fairlearn=fairlearn).items()}},
          open(OUT/"environment.json", "w"), indent=2)
''')

nb.cells = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "Paper1_Fairness_Audit.ipynb")
print("ok", len(C), "cells")
