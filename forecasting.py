import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# ── dados ──────────────────────────────────────────────────────────────────

URL = "https://raw.githubusercontent.com/mikemooreviz/superstore/master/superstore.csv"

df = pd.read_csv(URL)
df["Order Date"] = pd.to_datetime(df["Order Date"])

mensal = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
mensal.columns = ["mes", "vendas"]
mensal["mes"] = mensal["mes"].dt.to_timestamp()

# ── visualiza serie ─────────────────────────────────────────────────────────

mensal.plot(x="mes", y="vendas", figsize=(13, 4), title="Vendas mensais — Superstore")
plt.ylabel("Receita (USD)")
plt.tight_layout()
plt.savefig("img/serie_temporal.png", dpi=150)
plt.show()

# ── sazonalidade ────────────────────────────────────────────────────────────

mensal["mes_n"] = mensal["mes"].dt.month
mensal["ano"]   = mensal["mes"].dt.year

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

mensal.groupby("mes_n")["vendas"].mean().plot(kind="bar", ax=axes[0], color="#7c3aed")
axes[0].set_xticklabels(["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"], rotation=45)
axes[0].set_title("Receita média por mês")

mensal.groupby("ano")["vendas"].sum().plot(kind="bar", ax=axes[1], color="#16a34a")
axes[1].set_title("Receita total por ano")
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig("img/sazonalidade.png", dpi=150)
plt.show()

# ── features ────────────────────────────────────────────────────────────────

m = mensal.copy().sort_values("mes").reset_index(drop=True)

m["tri"] = m["mes"].dt.quarter
m["l1"]  = m["vendas"].shift(1)
m["l2"]  = m["vendas"].shift(2)
m["l3"]  = m["vendas"].shift(3)
m["ma3"] = m["vendas"].rolling(3).mean()
m["ma6"] = m["vendas"].rolling(6).mean()
m = m.dropna().reset_index(drop=True)

feats = ["mes_n", "tri", "ano", "l1", "l2", "l3", "ma3", "ma6"]
X, y  = m[feats], m["vendas"]

# ── treina modelos ──────────────────────────────────────────────────────────

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
lr = LinearRegression().fit(X_tr, y_tr)

print(f"{'Modelo':<20} {'MAE':>10} {'MAPE':>8}")
print("-" * 40)
for nome, mdl in [("Random Forest", rf), ("Gradient Boosting", gb), ("Linear Reg.", lr)]:
    p    = mdl.predict(X_te)
    mae  = mean_absolute_error(y_te, p)
    mape = mean_absolute_percentage_error(y_te, p) * 100
    print(f"{nome:<20} ${mae:>9,.0f} {mape:>7.1f}%")

# ── previsto vs real ────────────────────────────────────────────────────────

datas_te = m["mes"].iloc[-len(y_te):].values

plt.figure(figsize=(13, 5))
plt.plot(datas_te, y_te.values, color="black", linewidth=2, label="Real")
for nome, mdl, cor in [("Random Forest", rf, "#16a34a"),
                        ("Gradient Boosting", gb, "#dc2626"),
                        ("Linear Reg.", lr, "#2563eb")]:
    p    = mdl.predict(X_te)
    mape = mean_absolute_percentage_error(y_te, p) * 100
    plt.plot(datas_te, p, linestyle="--", linewidth=1.5, color=cor, label=f"{nome} (MAPE {mape:.1f}%)")

plt.title("Previsão vs Real")
plt.legend()
plt.tight_layout()
plt.savefig("img/previsao_vs_real.png", dpi=150)
plt.show()

# ── forecast 6 meses ────────────────────────────────────────────────────────

l1, l2, l3 = m["vendas"].iloc[-1], m["vendas"].iloc[-2], m["vendas"].iloc[-3]
ma3  = m["vendas"].iloc[-3:].mean()
ma6  = m["vendas"].iloc[-6:].mean()
base = m["mes"].max()

rows = []
for i in range(1, 7):
    d = base + pd.DateOffset(months=i)
    x = pd.DataFrame([{"mes_n": d.month, "tri": (d.month - 1) // 3 + 1, "ano": d.year,
                        "l1": l1, "l2": l2, "l3": l3, "ma3": ma3, "ma6": ma6}])
    p = rf.predict(x)[0]
    rows.append({"data": d, "forecast": round(p, 2)})
    l3, l2, l1 = l2, l1, p
    ma3 = (ma3 * 2 + p) / 3
    ma6 = (ma6 * 5 + p) / 6

fc = pd.DataFrame(rows)
print("\nForecast próximos 6 meses:")
print(fc.to_string(index=False))

# ── plot forecast ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(m["mes"], m["vendas"], color="black", linewidth=2, label="Histórico")
ax.plot(fc["data"], fc["forecast"], color="#2563eb", linewidth=2,
        linestyle="--", marker="o", label="Forecast (RF)")
ax.axvline(m["mes"].max(), color="gray", linestyle=":", linewidth=1)
ax.set_title("Forecast de vendas — próximos 6 meses")
ax.set_ylabel("Receita (USD)")
ax.legend()
plt.tight_layout()
plt.savefig("img/forecast.png", dpi=150)
plt.show()

# ── importancia das features ────────────────────────────────────────────────

imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
imp.plot(kind="barh", figsize=(8, 4), color="#7c3aed", title="Feature importance — Random Forest")
plt.tight_layout()
plt.savefig("img/feature_importance.png", dpi=150)
plt.show()
