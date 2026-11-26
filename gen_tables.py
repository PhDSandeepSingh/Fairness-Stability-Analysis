# Generates LaTeX tables and number macros for Paper 1 directly from results/raw_runs.csv
import pandas as pd, numpy as np, json
from scipy import stats

R = pd.read_csv("results/raw_runs.csv")
D = pd.read_csv("results/table_datasets.csv")
env = json.load(open("results/environment.json"))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(o):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def compute_deltas(ratio):
    base = R[R.method == "Baseline"].set_index(["dataset", "rep", "model"])
    rows = []
    for (ds, mdl, meth), g in R[~R.method.isin(["Baseline", "Baseline-Thr"])].groupby(["dataset", "model", "method"]):
        g = g.set_index("rep"); b = base.xs((ds, mdl), level=("dataset", "model")).loc[g.index]
        for met in ("dpd", "eod", "ba"):
            d = (g[met] - b[met]).values; n = len(d); m = d.mean(); sd = d.std(ddof=1)
            se = np.sqrt((1/n + ratio) * sd**2) if sd > 0 else np.nan
            t = m / se if se and se > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0
            h = stats.t.ppf(0.975, n - 1) * se if np.isfinite(se) else np.nan
            try: pw = stats.wilcoxon(d).pvalue
            except Exception: pw = 1.0
            pn = stats.ttest_1samp(d, 0).pvalue if sd > 0 else 1.0
            better = (d < 0) if met != "ba" else (d > 0)
            win = better.mean(); ties = int((d == 0).sum())
            z = 1.959963984540054; denom = 1 + z**2/n
            center = (win + z**2/(2*n))/denom; half = z*np.sqrt((win*(1-win) + z**2/(4*n))/n)/denom
            flip = ((np.sign(d) != np.sign(m)) & (d != 0)).mean()
            rows.append(dict(dataset=ds, model=mdl, method=meth, metric=met, n=n, mean_delta=m, sd_delta=sd,
                             ci_lo=m - h, ci_hi=m + h, p_nb=p, p_wilcoxon=pw, p_naive=pn,
                             win_rate=win, win_ci_lo=max(0, center-half), win_ci_hi=min(1, center+half),
                             ties=ties, sign_flip=flip, base_mean=b[met].mean(), base_sd=b[met].std(ddof=1)))
    T = pd.DataFrame(rows)
    for met in ("dpd", "eod", "ba"):
        mk = T.metric == met
        T.loc[mk, "p_holm"] = holm(T.loc[mk, "p_nb"].values)
        T.loc[mk, "p_naive_holm"] = holm(T.loc[mk, "p_naive"].values)
        T.loc[mk, "p_wilcoxon_holm"] = holm(T.loc[mk, "p_wilcoxon"].values)
        for ds in T.dataset.unique():
            mk2 = mk & (T.dataset == ds)
            T.loc[mk2, "p_holm_within"] = holm(T.loc[mk2, "p_nb"].values)
    def verdict(r):
        if r.metric == "ba": return ""
        if r.p_holm < 0.05: return "robust gain" if r.mean_delta < 0 else "robust harm"
        return "indistinguishable from noise"
    T["verdict"] = T.apply(verdict, axis=1)
    return T

def compute_sens(ratio):
    bt = R[R.method == "Baseline-Thr"].set_index(["dataset", "rep", "model"])
    srows = []
    for (ds, mdl, meth), g in R[R.method.str.startswith("ThreshOpt")].groupby(["dataset", "model", "method"]):
        g = g.set_index("rep"); b = bt.xs((ds, mdl), level=("dataset", "model")).loc[g.index]
        for met in ("ba", "dpd", "eod"):
            d = (g[met] - b[met]).values; n = len(d); m = d.mean(); sd = d.std(ddof=1)
            se = np.sqrt((1/n + ratio) * sd**2) if sd > 0 else np.nan
            t = m / se if se and se > 0 else np.nan
            srows.append(dict(dataset=ds, model=mdl, method=meth, metric=met, mean_delta=m,
                              p_nb=2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0))
    S = pd.DataFrame(srows)
    for met in ("ba", "dpd", "eod"):
        mk = S.metric == met; S.loc[mk, "p_holm"] = holm(S.loc[mk, "p_nb"].values)
    return S

T = compute_deltas(0.3 / 0.525)
S = compute_sens(0.3 / 0.525)
T07 = compute_deltas(0.3 / 0.7)
S07 = compute_sens(0.3 / 0.7)
T.to_csv("results/table_paired_deltas.csv", index=False)
S.to_csv("results/table_sensitivity_tuned_threshold.csv", index=False)
T07.to_csv("results/table_paired_deltas_ratio07.csv", index=False)

H = T[T.metric.isin(["dpd", "eod"])]
H07 = T07[T07.metric.isin(["dpd", "eod"])]
order = ["adult", "german", "bank", "credit"]; NICE = dict(adult="Adult", german="German Credit", bank="Bank Marketing", credit="Credit Default")
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
mac = {}
def M(k, v): mac[k] = v
M("NComp", str(H.groupby("metric").size()["dpd"])); M("NCompAll", str(len(H)))
M("NRep", str(env["n_repeats"])); M("NFits", str(len(R[R.method != "Baseline-Thr"])))

# Critical standardised effect (50% power) and conventional 80%-power MDE at first Holm step
t_crit = stats.t.ppf(1 - (0.05 / 44) / 2, int(env["n_repeats"]) - 1)
t_80 = stats.t.ppf(0.80, int(env["n_repeats"]) - 1)
fac525 = np.sqrt(1 / int(env["n_repeats"]) + 0.3 / 0.525)
fac07 = np.sqrt(1 / int(env["n_repeats"]) + 0.3 / 0.7)
crit525, mde525 = t_crit * fac525, (t_crit + t_80) * fac525
crit07, mde07 = t_crit * fac07, (t_crit + t_80) * fac07
M("CritPrimary", f2(crit525)); M("MdePrimary", f2(mde525))
M("CritOuter", f2(crit07)); M("MdeOuter", f2(mde07))
for ds, dtag in (("adult", "Adult"), ("bank", "Bank"), ("credit", "Credit"), ("german", "German")):
    for met, mtag in (("dpd", "DPD"), ("eod", "EOD")):
        med_sd = H[(H.dataset == ds) & (H.metric == met)].sd_delta.median()
        M(f"Mde{dtag}{mtag}", f3(med_sd * mde525))
        M(f"Crit{dtag}{mtag}", f3(med_sd * crit525))

for met, tag in (("dpd", "DPD"), ("eod", "EOD")):
    h = H[H.metric == met]; h07 = H07[H07.metric == met]
    M(f"Gain{tag}", str((h.verdict == "robust gain").sum())); M(f"Noise{tag}", str((h.verdict == "indistinguishable from noise").sum()))
    M(f"Harm{tag}", str((h.verdict == "robust harm").sum())); M(f"Naive{tag}", str((h.p_naive_holm < .05).sum()))
    M(f"Wilc{tag}", str((h.p_wilcoxon_holm < .05).sum())); M(f"Corr{tag}", str((h.p_holm < .05).sum()))
    M(f"Within{tag}", str((h.p_holm_within < .05).sum())); M(f"WithinOuter{tag}", str((h07.p_holm_within < .05).sum()))
    M(f"GainOuter{tag}", str((h07.verdict == "robust gain").sum()))
    M(f"Ties{tag}", str(int(h.ties.sum())))
    M(f"MedFlip{tag}", f2(h.sign_flip.median())); M(f"FlipTwenty{tag}", f"{100*(h.sign_flip >= .2).mean():.1f}")
    M(f"MaxFlip{tag}", f2(h.sign_flip.max()))
M("NaiveTotal", str((H.p_naive_holm < .05).sum())); M("WilcTotal", str((H.p_wilcoxon_holm < .05).sum()))
M("CorrTotal", str((H.p_holm < .05).sum())); M("CorrOuterTotal", str((H07.p_holm < .05).sum()))
M("NaiveOnly", str(((H.p_naive_holm < .05) & (H.p_holm >= .05)).sum()))
ns = H[H.verdict == "indistinguishable from noise"]
M("NNoise", str(len(ns))); M("NoiseWin", f2(ns.win_rate.mean())); M("NoiseWinHalf", f"{100*(ns.win_rate >= .5).mean():.1f}")
gd = D.set_index(D.dataset)
M("GermanMinority", str(int(gd.loc["German Credit", "n_g0"]))); M("GermanMinTest", str(int(round(gd.loc["German Credit", "n_g0"] * 0.3))))
ba05 = T[(T.metric == "ba") & T.method.str.startswith("ThreshOpt")].mean_delta.mean(); bat = S[S.metric == "ba"].mean_delta.mean()
M("TOBAvsHalf", f3(ba05)); M("TOBAvsTuned", f3(bat))
M("TOSigBAneg", str(((S.metric == "ba") & (S.p_holm < .05) & (S.mean_delta < 0)).sum())); M("TONSens", str((S.metric == "ba").sum()))
for met, tag in (("dpd", "DPD"), ("eod", "EOD")):
    s = S[S.metric == met]; s07 = S07[S07.metric == met]
    M(f"TOSigImp{tag}", str(((s.p_holm < .05) & (s.mean_delta < 0)).sum()))
    M(f"TOSigImpOuter{tag}", str(((s07.p_holm < .05) & (s07.mean_delta < 0)).sum()))
b = R[R.method.isin(["Baseline", "Baseline-Thr"])].groupby(["dataset", "method"])[["ba", "dpd", "eod"]].mean()
M("BankDPDhalf", f2(b.loc[("bank", "Baseline"), "dpd"])); M("BankDPDtuned", f2(b.loc[("bank", "Baseline-Thr"), "dpd"]))
M("BaseGermanDPD", f3(b.loc[("german", "Baseline"), "dpd"])); M("BaseGermanEOD", f3(b.loc[("german", "Baseline"), "eod"]))
M("AdultEGDPeod", f3(T[(T.dataset=="adult")&(T.method=="ExpGrad-DP")&(T.metric=="eod")].mean_delta.iloc[0]))
M("AdultFavMale", f3(gd.loc["Adult", "fav_rate_g1"])); M("AdultFavFemale", f3(gd.loc["Adult", "fav_rate_g0"]))
for k in ["sklearn", "fairlearn", "numpy", "pandas", "scipy"]: M("Ver" + k.capitalize(), env["packages"][k])
M("VerPython", env["python"])
with open("manuscript/numbers.tex", "w") as f:
    for k, v in mac.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# Table 1: datasets
bl = R[R.method == "Baseline"].groupby("dataset")[["ba", "dpd", "eod"]].agg(["mean", "std"])
L = [r"\begin{table}[t]", r"\centering", r"\caption{Benchmarks, sensitive attributes, and baseline behaviour (mean $\pm$ SD over 60 learner $\times$ split runs: 3 learners $\times$ " + mac["NRep"] + r" splits). Fav.\ = favourable outcome rate in Group 0 ($A{=}0$) and Group 1 ($A{=}1$); DPD = demographic parity difference; EOD = equalized odds difference.}", r"\label{tab:datasets}", r"\scriptsize", r"\begin{tabularx}{\textwidth}{@{}l@{\hspace{0.8em}}r@{\hspace{1em}}>{\raggedright\arraybackslash}X c c c c@{}}", r"\toprule",
     r"Dataset & $n$ & Attribute ($A{=}1$) & Fav.\ $A{=}0$ & Fav.\ $A{=}1$ & Base DPD & Base EOD \\", r"\midrule"]
SA = dict(adult="sex: male", german="age $\\ge 25$", bank="age 25--60", credit="sex: male")
for k in order:
    r = gd.loc[NICE[k]]
    L.append(f"{NICE[k]} & {int(r.n):,} & {SA[k]} & {f3(r.fav_rate_g0)} & {f3(r.fav_rate_g1)} & ${f3(bl.loc[k,('dpd','mean')])}\\pm{f3(bl.loc[k,('dpd','std')])}$ & ${f3(bl.loc[k,('eod','mean')])}\\pm{f3(bl.loc[k,('eod','std')])}$ \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' analysis of UCI Machine Learning Repository datasets~\cite{kohavi1996scaling,hofmann1994statlog,moro2014data,yeh2009comparisons}.", r"\end{table}"]
open("manuscript/tab_datasets.tex", "w").write("\n".join(L))

# Table 2: verdicts per dataset
L = [r"\begin{table}[t]", r"\centering", r"\caption{Audit verdicts for the " + mac["NComp"] + r" mitigation-versus-baseline comparisons per fairness metric (11 per dataset) using the primary Nadeau--Bengio ratio ($0.3/0.525$, Holm-adjusted across 44 comparisons, $\alpha=0.05$). DPD = demographic parity difference; EOD = equalized odds difference; G = robust gain; N = indistinguishable from noise; H = robust harm; Naive = naive paired $t$-test significant; Flip = median directional-reversal rate.}", r"\label{tab:verdicts}", r"\scriptsize",
     r"\begin{tabularx}{\textwidth}{@{}l *{10}{>{\centering\arraybackslash}X}@{}}", r"\toprule", r" & \multicolumn{5}{c}{DPD} & \multicolumn{5}{c}{EOD} \\ \cmidrule(lr){2-6}\cmidrule(l){7-11}", r"Dataset & G & N & H & Naive & Flip & G & N & H & Naive & Flip \\", r"\midrule"]
def row(h):
    out = []
    for met in ("dpd", "eod"):
        x = h[h.metric == met]
        out += [str((x.verdict == "robust gain").sum()), str((x.verdict.str.startswith("indist")).sum()), str((x.verdict == "robust harm").sum()), str((x.p_naive_holm < .05).sum()), f2(x.sign_flip.median())]
    return " & ".join(out)
for k in order: L.append(f"{NICE[k]} & {row(H[H.dataset == k])} \\\\")
L += [r"\midrule", f"All (0.3/0.525) & {row(H)} \\\\", r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' analysis of UCI benchmarks across 20 stratified splits.", r"\end{table}"]
open("manuscript/tab_verdicts.tex", "w").write("\n".join(L))

# Table 3: pooled-over-learner mean paired deltas per method
L = [r"\begin{table}[t]", r"\centering", r"\caption{Mean paired change relative to the same-learner baseline (negative DPD/EOD = fairer, positive BA = better utility). RW and TO are averaged over three learners (LR, RF, HGB); EG is evaluated with LR only. DPD = demographic parity difference; EOD = equalized odds difference; BA = balanced accuracy; RW = reweighing; TO = ThresholdOptimizer; EG = ExponentiatedGradient. Parentheses give the number of robust gains under the $0.3/0.525$ ratio (of 3 for RW/TO, of 1 for EG).}", r"\label{tab:deltas}", r"\scriptsize",
     r"\begin{tabularx}{\textwidth}{@{}l l *{3}{>{\raggedleft\arraybackslash}X} | *{2}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule",
     r" & & \multicolumn{3}{c|}{Three learners (LR, RF, HGB)} & \multicolumn{2}{c}{LR only} \\ \cmidrule(lr){3-5}\cmidrule(l){6-7}",
     r"Dataset & Metric & RW & TO-DP & TO-EO & EG-DP & EG-EO \\", r"\midrule"]
meths = ["Reweighing", "ThreshOpt-DP", "ThreshOpt-EO", "ExpGrad-DP", "ExpGrad-EO"]
for k in order:
    for i, (met, lab) in enumerate((("dpd", r"$\Delta$DPD"), ("eod", r"$\Delta$EOD"), ("ba", r"$\Delta$BA"))):
        cells = []
        for me in meths:
            x = T[(T.dataset == k) & (T.method == me) & (T.metric == met)]
            val = x.mean_delta.mean()
            if abs(val) < 0.0005: val = 0.0
            s_val = f"{val:+.3f}" if val != 0.0 else "0.000"
            v = f"${s_val}$ ({(x.verdict == 'robust gain').sum()})" if met != "ba" else f"${s_val}$"
            cells.append(v)
        L.append(f"{NICE[k] if i == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
    if k != order[-1]: L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' analysis of UCI benchmarks across 20 stratified splits.", r"\end{table}"]
open("manuscript/tab_deltas.tex", "w").write("\n".join(L))

# Table 4: sensitivity
L = [r"\begin{table}[t]", r"\centering", r"\caption{Post-hoc sensitivity analysis: ThresholdOptimizer (TO) compared with two reference baselines, averaged over three learners. BA = balanced accuracy; DPD = demographic parity difference; EOD = equalized odds difference; Ref.\ 0.5 = default decision threshold; Ref.\ tuned = one group-blind threshold tuned for BA on the post-processing split.}", r"\label{tab:sens}", r"\scriptsize",
     r"\begin{tabularx}{\textwidth}{@{}l l *{4}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule", r" & & \multicolumn{2}{c}{$\Delta$BA} & $\Delta$DPD & $\Delta$EOD \\ \cmidrule(lr){3-4}", r"Dataset & Method & ref.\ 0.5 & ref.\ tuned & ref.\ tuned & ref.\ tuned \\", r"\midrule"]
for k in order:
    for j, me in enumerate(["ThreshOpt-DP", "ThreshOpt-EO"]):
        a = T[(T.dataset == k) & (T.method == me) & (T.metric == "ba")].mean_delta.mean()
        s = S[(S.dataset == k) & (S.method == me)].groupby("metric").mean_delta.mean()
        L.append(f"{NICE[k] if j == 0 else ''} & {me.replace('ThreshOpt','TO')} & ${a:+.3f}$ & ${s['ba']:+.3f}$ & ${s['dpd']:+.3f}$ & ${s['eod']:+.3f}$ \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' analysis of UCI benchmarks across 20 stratified splits.", r"\end{table}"]
open("manuscript/tab_sens.tex", "w").write("\n".join(L))
print("P1 gen_tables.py R2 done:", mac["GainDPD"], mac["CritPrimary"], mac["MdePrimary"])
