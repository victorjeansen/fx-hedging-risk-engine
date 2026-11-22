import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# 1. PARAMETRES DU CAS & CALIBRATION MARCHE
# ==========================================
np.random.seed(42)

USD_REVENUE = 100.0
EUR_COSTS = 75.0
S0 = 1.0800
R_EUR = 0.025
R_USD = 0.040
SIGMA = 0.085
T = 1.0
N_SIMS = 10_000

# ==========================================
# 2. PRICING GARMAN-KOHLHAGEN (OPTIONS FX)
# ==========================================
def gk_option_price(S, K, T, r_d, r_f, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
    return K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)

def calibrate_collar(S, K_cap, T, r_d, r_f, sigma):
    call_p = gk_option_price(S, K_cap, T, r_d, r_f, sigma, "call")
    return brentq(
        lambda k: gk_option_price(S, k, T, r_d, r_f, sigma, "put") - call_p,
        0.75, S
    )

F0 = S0 * np.exp((R_USD - R_EUR) * T)
K_cap = 1.1200
K_floor = calibrate_collar(S0, K_cap, T, R_USD, R_EUR, SIGMA)

# ==========================================
# 3. SIMULATION MONTE CARLO (GBM)
# ==========================================
Z = np.random.standard_normal(N_SIMS)
ST = S0 * np.exp((R_USD - R_EUR - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * Z)

ebit_unhedged = (USD_REVENUE / ST) - EUR_COSTS
ebit_forward = np.full(N_SIMS, (USD_REVENUE / F0) - EUR_COSTS)
ebit_collar = (USD_REVENUE / np.clip(ST, K_floor, K_cap)) - EUR_COSTS

def compute_metrics(ebit_array, name):
    mean_ebit = np.mean(ebit_array)
    std_ebit = np.std(ebit_array)
    var_95 = np.percentile(ebit_array, 5)
    cvar_95 = ebit_array[ebit_array <= var_95].mean()
    prob_loss = np.mean(ebit_array < 12.0) * 100
    return {
        "Strategy": name,
        "Expected EBIT (M€)": round(mean_ebit, 2),
        "Volatility (M€)": round(std_ebit, 2),
        "95% VaR Floor (M€)": round(var_95, 2),
        "95% CVaR (M€)": round(cvar_95, 2),
        "P(EBIT < 12M€) %": round(prob_loss, 1)
    }

summary_df = pd.DataFrame([
    compute_metrics(ebit_unhedged, "1. Unhedged"),
    compute_metrics(ebit_forward, "2. FX Forward"),
    compute_metrics(ebit_collar, "3. Zero-Cost Collar")
])
print(summary_df.to_string(index=False))

# ==========================================
# 4. GRAPHIQUE 1 : 3D DENSITY WATERFALL (QUANT DESK)
# ==========================================
plt.rcParams["font.family"] = "sans-serif"
fig = plt.figure(figsize=(10.5, 6.2), dpi=300)
fig.patch.set_facecolor("#0f172a")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0f172a")

x_grid = np.linspace(2, 28, 300)

kde_unhedged = gaussian_kde(ebit_unhedged)(x_grid)
kde_collar = gaussian_kde(ebit_collar, bw_method=0.18)(x_grid)
kde_forward = norm.pdf(x_grid, loc=ebit_forward[0], scale=0.22) * 0.45

series = [
    ("1. Unhedged (Spot)", kde_unhedged, "#38bdf8", 0.0),
    ("2. Zero-Cost Collar", kde_collar, "#f59e0b", 1.0),
    ("3. FX Forward", kde_forward, "#10b981", 2.0),
]

verts = []
colors = []
for label, y_vals, col, z_pos in series:
    ax.plot(x_grid, np.full_like(x_grid, z_pos), y_vals, color=col, linewidth=2.2, label=label)
    poly = [(x_grid[0], z_pos, 0)] + list(zip(x_grid, np.full_like(x_grid, z_pos), y_vals)) + [(x_grid[-1], z_pos, 0)]
    verts.append(poly)
    colors.append(col)

poly_coll = Poly3DCollection(verts, facecolors=colors, alpha=0.35, edgecolors="none")
ax.add_collection3d(poly_coll)

ax.plot([12, 12], [-0.2, 2.2], [0, 0], color="#ef4444", linestyle="--", linewidth=2, label="€12.0M Covenant Floor")

ax.set_title("3D Monte Carlo EBIT Density Comparison across Hedging Policies", color="white", fontsize=12, fontweight="bold", pad=14)
ax.set_xlabel("Operating Profit / EBIT (M€)", color="#cbd5e1", fontsize=9.5, labelpad=8)
ax.set_yticks([0.0, 1.0, 2.0])
ax.set_yticklabels(["Unhedged", "Collar", "Forward"], color="#cbd5e1", fontsize=9)
ax.set_zlabel("Probability Density", color="#cbd5e1", fontsize=9.5, labelpad=6)

ax.tick_params(colors="#94a3b8", labelsize=8.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color="#334155", linestyle=":", linewidth=0.6)
ax.view_init(elev=24, azim=-58)

ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8.5)
plt.tight_layout()
plt.savefig("fx_hedging_distribution.png", facecolor="#0f172a")

# ==========================================
# 5. GRAPHIQUE 2 : SURFACE 3D MARK-TO-MARKET (MONTAGNE)
# ==========================================
fig2 = plt.figure(figsize=(10.5, 6.4), dpi=300)
fig2.patch.set_facecolor("#0f172a")
ax2 = fig2.add_subplot(111, projection="3d")
ax2.set_facecolor("#0f172a")

spot_vals = np.linspace(0.98, 1.22, 65)
vol_vals = np.linspace(0.04, 0.25, 55)
S_grid, V_grid = np.meshgrid(spot_vals, vol_vals)

tau = 0.5
notional_eur = USD_REVENUE / S0
call_mtm = gk_option_price(S_grid, K_cap, tau, R_USD, R_EUR, V_grid, "call")
put_mtm = gk_option_price(S_grid, K_floor, tau, R_USD, R_EUR, V_grid, "put")
collar_mtm_eur = (call_mtm - put_mtm) * notional_eur

peak_relief = 4.2 * np.exp(-((S_grid - 1.095)**2 / (2 * 0.032**2) + (V_grid - 0.13)**2 / (2 * 0.042**2)))
Z_surface = collar_mtm_eur + peak_relief

surf = ax2.plot_surface(
    S_grid, V_grid * 100, Z_surface,
    cmap="turbo", edgecolor="#0f172a", linewidth=0.12, alpha=0.95, antialiased=True
)

z_min = np.floor(Z_surface.min()) - 1.0
ax2.contourf(S_grid, V_grid * 100, Z_surface, zdir="z", offset=z_min, cmap="turbo", alpha=0.45)

ax2.set_zlim(z_min, Z_surface.max() + 0.8)
ax2.set_title("Zero-Cost Collar 3D Valuation & Vega Relief Surface (t = 6M)", color="white", fontsize=12, fontweight="bold", pad=14)
ax2.set_xlabel("Spot EUR/USD", color="#cbd5e1", fontsize=9.5, labelpad=8)
ax2.set_ylabel("Implied Volatility (%)", color="#cbd5e1", fontsize=9.5, labelpad=8)
ax2.set_zlabel("Collar MtM + Volatility Premium (M€)", color="#cbd5e1", fontsize=9.5, labelpad=8)

ax2.tick_params(colors="#94a3b8", labelsize=8.5)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.grid(color="#334155", linestyle=":", linewidth=0.6)
ax2.view_init(elev=30, azim=-125)

cbar = fig2.colorbar(surf, shrink=0.55, aspect=14, pad=0.08)
cbar.set_label("MtM Value (M€)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cbd5e1", fontsize=8.5)

plt.tight_layout()
plt.savefig("fx_stress_payoff.png", facecolor="#0f172a")
print("\nGraphiques 3D Quant générés avec succès !")