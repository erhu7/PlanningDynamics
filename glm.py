import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── 0. Prepare data ───────────────────────────────────────────────────────────
SUBJECT = 'bart'
#TARGET = 'steps_extra'  # or 'plannings'
CATEGORICAL_FEATURES = ['steps_teleporter_saved', 'preferred']
FEATURES = ['shortest_paths', 
            'shortest_path_length', 
            'steps_extra', 
            'teleporter_used', 
            'plannings', 
            'progression',
            'steps_teleporter_saved',
            'preferred',]

#subject_initial = SUBJECT[0].upper()
df = pd.read_pickle('behaviordata_v4.pkl')
df = df[df['subject'] == SUBJECT].copy()   

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype('category')
    df[col] = pd.Categorical(df[col], categories=sorted(df[col].unique()))

# ── 1. Build design matrices (for discrete models) ───────────────────────────
def build_design_matrix(df, response):
    features = [col for col in FEATURES if col != response]
    X = pd.get_dummies(
        df[features],
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=int
    )

    y = df[response].values
    return sm.add_constant(X), X, y, X.columns.tolist()

X_extra_c, X_extra, y_extra, feat_extra = build_design_matrix(
    df, response='steps_extra')

X_plan_c, X_plan, y_plan, feat_plan = build_design_matrix(
    df, response='plannings')

# ── 2. VIF ────────────────────────────────────────────────────────────────────
def compute_vif(X, feature_names):
    vif_df = pd.DataFrame({
        'Feature': feature_names,
        'VIF': [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False)
    return vif_df

print("=== VIF: Steps_extra model ===")
print(compute_vif(X_extra, feat_extra).to_string(index=False))

print("\n=== VIF: plannings model ===")
print(compute_vif(X_plan, feat_plan).to_string(index=False))

print("\n=== Check for near-zero variance columns (can break Hessian) ===")
print(X_extra.std().sort_values())

print("\n=== Check condition number (> 1000 → severe multicollinearity) ===")
print(f"Condition number: {np.linalg.cond(X_extra_c):.1f}")

print("\n=== Check for any column that is nearly constant ===")
print((X_extra == 0).mean().sort_values())

# ── 3. Model fitting functions ────────────────────────────────────────────────
def fit_gaussian(y_train, X_train, X_test):
    m = sm.GLM(y_train, X_train,
               family=sm.families.Gaussian(sm.families.links.Identity())
               ).fit()
    return m, m.predict(X_test)

def fit_nb(y_train, X_train, X_test):
    m = NegativeBinomial(y_train, X_train).fit(disp=0)
    return m, m.predict(X_test)

def fit_zinb(y_train, X_train, X_test):
    infl_train = np.ones((len(y_train), 1))
    infl_test  = np.ones((len(X_test),  1))
    m = ZeroInflatedNegativeBinomialP(
            y_train, X_train, exog_infl=infl_train
        ).fit(method='nm', maxiter=2000, disp=0)
    preds = m.predict(X_test, exog_infl=infl_test)
    return m, preds

# ── 4. Full-data fits (for summary & AIC) ────────────────────────────────────
print("\nFitting full models...")

# Steps_extra
gauss_extra_full, _ = fit_gaussian(y_extra, X_extra_c, X_extra_c)
zinb_extra_full,  _ = fit_zinb(y_extra, X_extra_c, X_extra_c)

# plannings
gauss_plan_full, _  = fit_gaussian(y_plan, X_plan_c, X_plan_c)
nb_plan_full,    _  = fit_nb(y_plan, X_plan_c, X_plan_c)

# ── 5. AIC / BIC comparison ───────────────────────────────────────────────────
def model_comparison_table(models_dict):
    rows = []
    for name, m in models_dict.items():
        rows.append({
            'Model':    name,
            'AIC':      round(m.aic, 1),
            'BIC':      round(m.bic, 1),
            'Log-Lik':  round(m.llf, 1)
        })
    df_comp = pd.DataFrame(rows).sort_values('AIC')
    df_comp['ΔAIC'] = (df_comp['AIC'] - df_comp['AIC'].min()).round(1)
    return df_comp

print("\n=== Model comparison: Steps_extra ===")
comp_extra = model_comparison_table({
    'Gaussian': gauss_extra_full,
    'ZINB':     zinb_extra_full
})
print(comp_extra.to_string(index=False))

print("\n=== Model comparison: plannings ===")
comp_plan = model_comparison_table({
    'Gaussian': gauss_plan_full,
    'NB2':      nb_plan_full
})
print(comp_plan.to_string(index=False))

# ── 6. Coefficient summary ────────────────────────────────────────────────────
def model_summary_df(result, feature_names):
    coef  = result.params
    pvals = result.pvalues
    ci    = result.conf_int()
    n     = min(len(coef), len(feature_names))
    return pd.DataFrame({
        'Feature': feature_names[:n],
        'Coef':    coef.values[:n],
        'CI_low':  ci.iloc[:n, 0].values,
        'CI_high': ci.iloc[:n, 1].values,
        'p_value': pvals.values[:n],
        'Sig':     pd.cut(pvals.values[:n],
                          bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                          labels=['***', '**', '*', ''])
    })

all_feat_extra = ['const'] + feat_extra
all_feat_plan  = ['const'] + feat_plan

print("\n=== ZINB summary: Steps_extra ===")
print(model_summary_df(zinb_extra_full, all_feat_extra).to_string(index=False))

print("\n=== NB2 summary: plannings ===")
print(model_summary_df(nb_plan_full, all_feat_plan).to_string(index=False))

'''
# ── 7. 5-fold Cross-Validation ────────────────────────────────────────────────
def cross_validate(y, X, fit_fn, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_list, mae_list = [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        try:
            _, preds = fit_fn(y[tr], X[tr], X[te])
            preds = np.clip(preds, 0, None)   # floor negatives for Gaussian
            rmse_list.append(np.sqrt(mean_squared_error(y[te], preds)))
            mae_list.append(np.mean(np.abs(y[te] - preds)))
        except Exception as e:
            print(f"  Fold {fold+1} failed: {e}")
    return np.array(rmse_list), np.array(mae_list)

cv_results = {}

print("\n=== 5-fold CV results ===")
for label, y, X, fit_fn in [
    ('Steps_extra | Gaussian', y_extra, X_extra_c, fit_gaussian),
    ('Steps_extra | ZINB',     y_extra, X_extra_c, fit_zinb),
    ('plannings   | Gaussian', y_plan,  X_plan_c,  fit_gaussian),
    ('plannings   | NB2',      y_plan,  X_plan_c,  fit_nb),
]:
    rmse, mae = cross_validate(y, X, fit_fn)
    cv_results[label] = {'RMSE_mean': rmse.mean(), 'RMSE_std': rmse.std(),
                          'MAE_mean':  mae.mean(),  'MAE_std':  mae.std()}
    print(f"  {label:30s} | RMSE: {rmse.mean():.3f} ± {rmse.std():.3f}"
          f"  MAE: {mae.mean():.3f} ± {mae.std():.3f}")

# ── 8. Diagnostic & comparison plots ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

plot_configs = [
    # (row, best_model,        gaussian_model,   y,       label,          feat_names)
    (0, zinb_extra_full,  gauss_extra_full, y_extra, 'Steps_extra', all_feat_extra),
    (1, nb_plan_full,     gauss_plan_full,  y_plan,  'plannings',   all_feat_plan),
]

cv_labels_map = {
    0: ('Steps_extra | ZINB', 'Steps_extra | Gaussian'),
    1: ('plannings   | NB2',  'plannings   | Gaussian'),
}

for row, best_m, gauss_m, y_obs, label, feat in plot_configs:

    fitted_best  = best_m.fittedvalues
    fitted_gauss = gauss_m.fittedvalues

    # ── Col 0: Residuals vs Fitted (best model) ──
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.scatter(fitted_best, y_obs - fitted_best, alpha=0.2, s=8, color='steelblue')
    ax0.axhline(0, color='red', lw=1.5)
    ax0.set_xlabel('Fitted'); ax0.set_ylabel('Residuals')
    ax0.set_title(f'{label}\nResiduals vs Fitted (best model)')

    # ── Col 1: Observed vs Predicted — both models overlaid ──
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.scatter(y_obs, fitted_best,  alpha=0.15, s=8,
                color='steelblue', label='ZINB/NB2')
    ax1.scatter(y_obs, fitted_gauss, alpha=0.15, s=8,
                color='salmon',    label='Gaussian')
    lim = max(y_obs.max(), fitted_best.max(), fitted_gauss.max())
    ax1.plot([0, lim], [0, lim], 'k--', lw=1)
    ax1.set_xlabel('Observed'); ax1.set_ylabel('Predicted')
    ax1.set_title('Observed vs Predicted')
    ax1.legend(fontsize=8)

    # ── Col 2: CV RMSE comparison (bar plot) ──
    ax2 = fig.add_subplot(gs[row, 2])
    best_key, gauss_key = cv_labels_map[row]
    names  = ['Gaussian', 'ZINB/NB2']
    means  = [cv_results[gauss_key]['RMSE_mean'],
              cv_results[best_key]['RMSE_mean']]
    stds   = [cv_results[gauss_key]['RMSE_std'],
              cv_results[best_key]['RMSE_std']]
    colors = ['salmon', 'steelblue']
    bars   = ax2.bar(names, means, yerr=stds, capsize=6,
                     color=colors, edgecolor='black', width=0.5)
    ax2.set_ylabel('CV RMSE'); ax2.set_title('5-fold CV RMSE comparison')
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds)*0.1,
                 f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    # ── Col 3: Coefficient plot (best model, exclude const) ──
    ax3 = fig.add_subplot(gs[row, 3])
    sumdf = model_summary_df(best_m, feat).query("Feature != 'const'")
    colors_coef = ['#d62728' if p < 0.05 else '#aec7e8'
                   for p in sumdf['p_value']]
    ax3.barh(sumdf['Feature'], sumdf['Coef'],
             xerr=[sumdf['Coef'] - sumdf['CI_low'],
                   sumdf['CI_high'] - sumdf['Coef']],
             color=colors_coef, capsize=4, edgecolor='black', height=0.6)
    ax3.axvline(0, color='black', lw=1)
    ax3.set_title('Coefficients\n(red = p < 0.05)')
    ax3.set_xlabel('Coefficient')

plt.suptitle('GLM Diagnostics: ZINB/NB2 vs Gaussian Baseline', 
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('glm_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
'''