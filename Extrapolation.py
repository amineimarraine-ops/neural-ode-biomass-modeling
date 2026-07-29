#Imports
import torch
import random
import os
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr
import optax
from tqdm import tqdm

path = os.getcwd() + "/ambr"
files = os.listdir(path)
files_xls = [f for f in files if (f[-3:] == 'csv') and f[8] == '1']
df = pd.DataFrame()
files_xls

df = [None] * len(files_xls)

for f in range(len(files_xls)):
    data = pd.read_csv(path + "/" + files_xls[f])

    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace('\ufeff', '', regex=False)

    if "Time" not in data.columns:
        print(files_xls[f], data.columns.tolist())
        raise ValueError(f"No Time column in {files_xls[f]}")

    # Extract run tag from filename
    run_name = files_xls[f].replace(".csv", "")

    df[f] = data
    df[f] = df[f].loc[:, ~df[f].columns.duplicated()]
    df[f] = df[f].sort_values("Time").reset_index(drop=True)
    df[f] = df[f].drop(columns=[c for c in df[f].columns
             if ('Optical density' not in c) 
             and ('DO' not in c) 
             and ('pH' not in c) 
             and ('Time' not in c)])
run1_19_24 = ["Time" if x == "Time" else x + " run1" for x in df[0].columns]
run1_13_18 = ["Time" if x == "Time" else x + " run1" for x in df[1].columns]

# List of lists, one per dataframe
new_names = [run1_19_24, run1_13_18]

# Now assign
for d in range(len(df)):
    df[d].columns = new_names[d]
df_merged = df[0].copy()

for d in range(1, len(df)):
    df_merged = pd.merge(df_merged, df[d], on="Time", how="outer")

df_merged = df_merged.sort_values("Time").reset_index(drop=True)
path = os.getcwd() + "/Lorena_dataset/Glucose_files_310326/OfflineMetabolites_AMBR1.csv"
glucose1 = pd.read_csv(path, delimiter=';')
glc1 = glucose1[["Reactor","Time","Glucose"]]
col_to_drop = ["Bioreactor 13 - Optical density run1", "Bioreactor 13 - pH run1", "Bioreactor 13 - DO run1",
               "Bioreactor 19 - Optical density run1", "Bioreactor 19 - pH run1", "Bioreactor 19 - DO run1"]

df_merged = df_merged.drop(columns=col_to_drop)
glucose_df = pd.DataFrame()
bior = None

for i in range(1, len(df_merged.columns), 3):
    col = df_merged.columns[i]
    bio_name = col[:13]
    run_tag = col[-4:]

    if "run1" in col:
        bior = glc1[glc1["Reactor"] == bio_name][["Time", "Glucose"]].copy()
        bior = bior.rename(columns={"Glucose": f"{bio_name} {run_tag}"})

    # Rename Glucose column
    bior = bior.rename(columns={"Glucose": f"{bio_name} {run_tag}"})

    # Merge into glucose_df
    if glucose_df.empty:
        glucose_df = bior
    else:
        glucose_df = pd.merge(glucose_df, bior, on="Time", how="outer")

glucose_df = glucose_df.sort_values("Time").reset_index(drop=True)
cols = list(df_merged.columns)
for i, col in enumerate(cols):
    if "Optical density" not in col:
        continue
    new_col_name = col.replace("Optical density", "Biomass")
    ph_col = col.replace("Optical density", "pH")
    ph_idx = df_merged.columns.get_loc(ph_col)
    df_merged.insert(ph_idx + 1, new_col_name, df_merged[col] * 0.4267)
df_merged = df_merged.drop(columns=[c for c in df_merged.columns
             if ('Optical' in c)])

def normalize(ys, stats):
    result = []
    for i, y in enumerate(ys):
        var = i % 4
        key = ["glucose", "od", "ph", "biomass"][var]
        vmin, vmax = stats[key]
        result.append((y - vmin) / (vmax - vmin + 1e-8))
    return result

class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.out_scale * self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            adjoint=diffrax.DirectAdjoint(),
            max_steps=1000000,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


lr = 1e-4 #Do not change
width_size = 17
depth = 5

def main(variable_data_list, variable_ts_list, n_epochs, seed=5678):

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    data_size = 4

    model = NeuralODE(data_size, width_size, depth, key=model_key)
    optim = optax.adabelief(lr)
    """optim = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr)
)"""
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = model(ti, yi[0])  # shape (T, data_size)
        sigma = 1.0
        loss_classic = jnp.mean(((yi - y_pred) / sigma) ** 2)
        negativity_loss = jnp.mean(jnp.maximum(0.0, -y_pred) ** 2)
        return loss_classic + negativity_loss

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    loss_l = []
    n_vars = 4
    R2_scores = [[] for _ in range(n_vars)]

    # number of bioreactors = total entries / 4 variables each
    o = len(variable_data_list) // 4

    for epoch in tqdm(range(n_epochs)):
        idx = epoch % o

        # Get the 4 variables for this bioreactor
        glc_ys  = variable_data_list[4*idx]
        od_ys   = variable_data_list[4*idx + 1]
        ph_ys   = variable_data_list[4*idx + 2]
        bio_ys  = variable_data_list[4*idx + 3]

        # Get the corresponding time arrays
        glc_ts  = variable_ts_list[4*idx]
        od_ts   = variable_ts_list[4*idx + 1]
        ph_ts   = variable_ts_list[4*idx + 2]
        bio_ts  = variable_ts_list[4*idx + 3]

        # Use OD timestamps as the common time grid
        t = jnp.asarray(od_ts, dtype=jnp.float32)

        # Interpolate all variables onto the OD time grid
        ys = jnp.stack([
            jnp.interp(t, jnp.asarray(glc_ts), jnp.asarray(glc_ys)),  # glucose
            jnp.asarray(od_ys,  dtype=jnp.float32),                    # OD (reference)
            jnp.interp(t, jnp.asarray(ph_ts),  jnp.asarray(ph_ys)),   # pH
            jnp.interp(t, jnp.asarray(bio_ts), jnp.asarray(bio_ys)),  # biomass
        ], axis=1)  # shape (T, 4)

        y_true = torch.tensor(np.asarray(ys), dtype=torch.float32)

        loss, model, opt_state = make_step(t, ys, model, opt_state)

        y_pred = torch.tensor(np.asarray(model(t, ys[0])), dtype=torch.float32)

        loss_l.append(float(loss))

        for var_idx in range(n_vars):
            ss_res = torch.sum((y_true[:, var_idx] - y_pred[:, var_idx]) ** 2)
            ss_tot = torch.sum((y_true[:, var_idx] - torch.mean(y_true[:, var_idx])) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            R2_scores[var_idx].append(float(r2))

    return loss_l, R2_scores, t, ys, model

extrap_values = np.array([])

for i in range(8,0,-1):
    prct = 10
    ts_list = []
    ys_list = []
    ts_list_full = []
    ys_list_full = []

    for o in range(1, len(glucose_df.columns)):
        # --- GLUCOSE ---
        s = glucose_df[glucose_df.columns[i]].dropna()
        s_full = s.copy()
        indices = s.index[prct*len(s)//10:]
        s = s.drop(indices)
        ts      = glucose_df.loc[s.index, "Time"]
        ts_full = glucose_df.loc[s_full.index, "Time"]
        ts_list.append(ts.to_numpy(dtype=np.float32))
        ys_list.append(s.to_numpy(dtype=np.float32))
        ts_list_full.append(ts_full.to_numpy(dtype=np.float32))
        ys_list_full.append(s_full.to_numpy(dtype=np.float32))

        # --- OD, pH, Biomass ---
        s1 = df_merged[df_merged.columns[1:][3*i-3:3*i][0]].dropna()
        s2 = df_merged[df_merged.columns[1:][3*i-3:3*i][1]].dropna()
        s3 = df_merged[df_merged.columns[1:][3*i-3:3*i][2]].dropna()
        
        s1_full, s2_full, s3_full = s1.copy(), s2.copy(), s3.copy()

        indices_1 = s1.index[prct*len(s1)//10:]
        indices_2 = s2.index[prct*len(s2)//10:]
        indices_3 = s3.index[prct*len(s3)//10:]

        s1 = s1.drop(indices_1)
        s2 = s2.drop(indices_2)
        s3 = s3.drop(indices_3)

        ts1 = df_merged.loc[s1.index, "Time"]
        ts2 = df_merged.loc[s2.index, "Time"]
        ts3 = df_merged.loc[s3.index, "Time"]

        ts1_full = df_merged.loc[s1_full.index, "Time"]
        ts2_full = df_merged.loc[s2_full.index, "Time"]
        ts3_full = df_merged.loc[s3_full.index, "Time"]

        ts_list.extend([ts1.to_numpy(dtype=np.float32), ts2.to_numpy(dtype=np.float32), ts3.to_numpy(dtype=np.float32)])
        ys_list.extend([s1.to_numpy(dtype=np.float32),  s2.to_numpy(dtype=np.float32),  s3.to_numpy(dtype=np.float32)])
        ts_list_full.extend([ts1_full.to_numpy(dtype=np.float32), ts2_full.to_numpy(dtype=np.float32), ts3_full.to_numpy(dtype=np.float32)])
        ys_list_full.extend([s1_full.to_numpy(dtype=np.float32),  s2_full.to_numpy(dtype=np.float32),  s3_full.to_numpy(dtype=np.float32)])

    # --- Train/test split (on raw data, before normalization) ---
    list_bior = [i[11:13] for i in glucose_df.columns[1:]]
    test_bior = len(list_bior) - 2
    start = 4 * test_bior
    stop  = 4 * test_bior + 4

    train_raw   = ys_list[:start] + ys_list[stop:]
    train_t     = ts_list[:start] + ts_list[stop:]
    test_t      = ts_list_full[start:stop]

    # --- Compute normalization stats on training split ONLY ---
    glucose_train = np.concatenate(train_raw[0::4])
    od_train      = np.concatenate(train_raw[1::4])
    ph_train      = np.concatenate(train_raw[2::4])
    biomass_train = np.concatenate(train_raw[3::4])

    """stats = {
        "glucose": (glucose_train.mean(), glucose_train.std()),
        "od":      (od_train.mean(),      od_train.std()),
        "ph":      (ph_train.mean(),      ph_train.std()),
        "biomass": (biomass_train.mean(), biomass_train.std()),
    }
    """

    stats = {
        "glucose": (glucose_train.min(), glucose_train.max()),
        "od":      (od_train.min(),      od_train.max()),
        "ph":      (0,14),
        "biomass": (biomass_train.min(), biomass_train.max()),
    }

    # Normalize train (cropped) and test (full) using train stats only
    train = normalize(train_raw, stats)
    test  = normalize(ys_list_full[start:stop], stats)

    # Run
    n_epochs = 100000  # Max is 240000
    loss, R, t, ys, model = main(train, train_t, n_epochs)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    axs[0].plot(loss, color='black', linewidth=2)
    axs[0].grid(True)
    axs[0].set_title(f"LR = {lr} | Width = {width_size} | Depth = {depth}")
    axs[0].set_xlabel('Epochs', fontsize=15)
    axs[0].set_ylabel('Loss', fontsize=15)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    start_r = int(n_epochs * 0.96)  # zoom into the last 10% of training
    t_r = np.arange(start_r, n_epochs, 1)

    axs[1].plot(R[0])
    axs[1].grid(True)
    axs[1].set_title(f"LR = {lr} | Width = {width_size} | Depth = {depth}")
    axs[1].set_xlabel('Epochs', fontsize=15)
    axs[1].set_ylabel('R2 score', fontsize=15)

    axs[2].plot(t_r, R[0][start_r:])
    axs[2].grid(True)
    axs[2].set_title(f"LR = {lr} | Width = {width_size} | Depth = {depth}")
    axs[2].set_xlabel('Epochs', fontsize=15)
    axs[2].set_ylabel('R2 score (zoom)', fontsize=15)

    plt.tight_layout()
    fig_name = '/shared/projects/pinn_proj/scripts/extrapolation/loss_' + str(list_bior[test_bior])
    plt.savefig(fig_name+'.png', format='png') ###########################################

    # -----------------------------------
    # Build the COMPLETE test trajectory
    # -----------------------------------
    glc_ys = test[0]
    od_ys  = test[1]
    ph_ys  = test[2]
    bio_ys = test[3]

    glc_ts = test_t[0]
    od_ts  = test_t[1]
    ph_ts  = test_t[2]
    bio_ts = test_t[3]

    # Common full time grid
    t_test = jnp.asarray(od_ts, dtype=jnp.float32)

    # Align every variable to that common grid
    ys_test = jnp.stack([
        jnp.interp(
            t_test,
            jnp.asarray(glc_ts, dtype=jnp.float32),
            jnp.asarray(glc_ys, dtype=jnp.float32),
        ),
        jnp.interp(
            t_test,
            jnp.asarray(od_ts, dtype=jnp.float32),
            jnp.asarray(od_ys, dtype=jnp.float32),
        ),
        jnp.interp(
            t_test,
            jnp.asarray(ph_ts, dtype=jnp.float32),
            jnp.asarray(ph_ys, dtype=jnp.float32),
        ),
        jnp.interp(
            t_test,
            jnp.asarray(bio_ts, dtype=jnp.float32),
            jnp.asarray(bio_ys, dtype=jnp.float32),
        ),
    ], axis=1)

    # -----------------------------------
    # Define the cutoff on the test reactor
    # -----------------------------------
    cut_off = len(t_test) * i // 10
    cut_off = min(max(cut_off, 0), len(t_test) - 2)

    t_cutoff = t_test[cut_off]

    # Neural ODE should usually receive relative times starting at zero
    t_extrap = t_test[cut_off:] - t_cutoff

    # Measured state at cutoff becomes the new initial condition
    y0_extrap = ys_test[cut_off]

    # Ground truth for forecast interval only
    y_true_extrap_jax = ys_test[cut_off:]

    # Forecast only the remaining trajectory
    y_pred_extrap_jax = model(t_extrap, y0_extrap)

    assert y_true_extrap_jax.shape == y_pred_extrap_jax.shape, (
        f"Target shape {y_true_extrap_jax.shape}, "
        f"prediction shape {y_pred_extrap_jax.shape}"
    )

    # Convert to torch for metrics
    y_true_extrap = torch.tensor(
        np.asarray(y_true_extrap_jax),
        dtype=torch.float32,
    )

    y_pred_extrap = torch.tensor(
        np.asarray(y_pred_extrap_jax),
        dtype=torch.float32,
    )

    var_names = ["Glucose", "DO", "pH", "Biomass"]
    units = ["mol/L", "", "", "g/L"]

    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    for var_idx, name in enumerate(var_names):
        true_var = y_true_extrap[:, var_idx]
        pred_var = y_pred_extrap[:, var_idx]

        ss_res = torch.sum((true_var - pred_var) ** 2)
        ss_tot = torch.sum((true_var - torch.mean(true_var)) ** 2)
        r2_extrap = 1 - ss_res / (ss_tot + 1e-12)

        axs[var_idx].plot(
            t_test,
            ys_test[:, var_idx],
            label="Ground truth",
            linewidth=2,
        )

        axs[var_idx].plot(
            t_test[cut_off:],
            y_pred_extrap_jax[:, var_idx],
            label="Forecast",
            linewidth=2,
            linestyle="--",
        )

        axs[var_idx].axvline(
            x=float(t_cutoff),
            linestyle=":",
            linewidth=1.5,
            label="Forecast start",
        )

        axs[var_idx].set_title(
            f"Bioreactor {list_bior[test_bior]} - {name}"
        )
        axs[var_idx].set_xlabel("Time (h)")
        axs[var_idx].set_ylabel(units[var_idx])
        axs[var_idx].grid(True)
        axs[var_idx].legend()

        axs[var_idx].text(
            0.60,
            0.92,
            f"R² forecast = {float(r2_extrap):.3f}",
            transform=axs[var_idx].transAxes,
            fontsize=9,
        )

    plt.tight_layout()
    fig_name = '/shared/projects/pinn_proj/scripts/extrapolation/results_' + str(list_bior[test_bior])
    plt.savefig(fig_name+'.png', format='png') ###########################################
    
    var_names = ["Glucose", "OD", "pH", "Biomass"]
    for var_idx, name in enumerate(var_names):
        true_var = y_true_extrap[:, var_idx]
        pred_var = y_pred_extrap[:, var_idx]

        ss_res = torch.sum((true_var - pred_var) ** 2)
        ss_tot = torch.sum((true_var - torch.mean(true_var)) ** 2)

        r2 = 1 - ss_res / (ss_tot + 1e-12)
        mae = torch.mean(torch.abs(true_var - pred_var))
        rmse = torch.sqrt(torch.mean((true_var - pred_var) ** 2))
        nrmse = rmse / (torch.max(true_var) - torch.min(pred_var) + 1e-12)

        #print(f"R² {name}: {r2:.4f}")
        #print(f"RMSE {name}: {rmse:.4f}")
        #print(f"NRMSE {name}: {nrmse:.4f}")
        #print(f"MAE {name}: {mae:.4f}")

        extrap_values = np.append(extrap_values, max(r2, 0))
        extrap_values = np.append(extrap_values, max(rmse, 0))
        extrap_values = np.append(extrap_values, max(nrmse, 0))
        extrap_values = np.append(extrap_values, max(mae, 0))

x = np.arange(0.8, 0.1, -0.1)  # [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

var_names  = ["Glucose", "DO", "pH", "Biomass"]
val_names  = ["R²", "RMSE", "NRMSE", "MAE"]

# extrap_values structure per prct block (16 values):
# [Glc_R2, Glc_RMSE, Glc_NRMSE, Glc_MAE,
#  DO_R2,  DO_RMSE,  DO_NRMSE,  DO_MAE,
#  pH_R2,  pH_RMSE,  pH_NRMSE,  pH_MAE,
#  Bio_R2, Bio_RMSE, Bio_NRMSE, Bio_MAE]

for metric_idx, metric_name in enumerate(val_names):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f"Forecast {metric_name} vs observed fraction", fontsize=16)

    for var_idx, (var_name, ax) in enumerate(zip(var_names, axs.flatten())):
        # For each prct block, the value is at: block*16 + var_idx*4 + metric_idx
        values = [extrap_values[block*16 + var_idx*4 + metric_idx] for block in range(7)]

        ax.plot(x, values, color='black', linewidth=2, marker='o')
        ax.grid(True)
        ax.set_title(var_name)
        ax.set_xlabel('Observed trajectory fraction', fontsize=13)
        ax.set_ylabel(f'Forecast {metric_name}', fontsize=13)

    plt.tight_layout()
    fig_name = '/shared/projects/pinn_proj/scripts/extrapolation/results_' + str(metric_name)
    plt.savefig(fig_name+'.png', format='png') ###########################################
