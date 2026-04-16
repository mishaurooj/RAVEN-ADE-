import os
import json
import time
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

ATTACK_ORDER = ['Normal', 'DoS', 'Payload', 'HeaderFlood', 'APIAbuse', 'InsiderMimic']
ATTACK_COLORS = {
    'Normal': '#1f77b4',
    'DoS': '#d62728',
    'Payload': '#ff7f0e',
    'HeaderFlood': '#2ca02c',
    'APIAbuse': '#9467bd',
    'InsiderMimic': '#8c564b',
}


@dataclass
class PreparedData:
    all_features: List[str]
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train_hg: np.ndarray
    X_val_hg: np.ndarray
    X_test_hg: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    attack_train: np.ndarray
    attack_val: np.ndarray
    attack_test: np.ndarray
    scaler: StandardScaler


class VAESampling(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(self.beta * tf.reduce_mean(kl_loss))
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'beta': self.beta})
        return cfg


def set_pub_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 200,
        'savefig.dpi': 900,
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.18,
        'grid.linestyle': '--',
        'grid.linewidth': 0.45,
        'legend.frameon': False,
        'legend.fontsize': 6.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'lines.linewidth': 1.1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def apply_axis_style(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6.5, length=2.5, pad=1.5)
    ax.grid(True, alpha=0.16, linestyle='--', linewidth=0.4)


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.drop_duplicates().reset_index(drop=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['dayofweek'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
        df = df.drop(columns=['timestamp'])
    return df


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = [
        c for c in ['method', 'path', 'http_version', 'application_number', 'request_sequence', 'src_ip']
        if c in df.columns
    ]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def inject_attacks_with_types(df: pd.DataFrame) -> pd.DataFrame:
    normal_df = df.copy()
    normal_df['label'] = 0
    normal_df['attack_type'] = 'Normal'
    n_normal = len(normal_df)
    attack_per_type = n_normal // 5

    def inject(base_df: pd.DataFrame, n: int, mod_fn, name: str) -> pd.DataFrame:
        s = base_df.sample(n=min(n, len(base_df)), replace=True, random_state=42).copy()
        mod_fn(s)
        s['label'] = 1
        s['attack_type'] = name
        return s

    def dos_mod(s: pd.DataFrame) -> None:
        if 'processing_time_ms' in s.columns:
            s['processing_time_ms'] *= np.random.uniform(8, 15, len(s))

    def payload_mod(s: pd.DataFrame) -> None:
        if 'vars_size_bytes' in s.columns:
            s['vars_size_bytes'] *= np.random.uniform(10, 20, len(s))
        if 'num_vars' in s.columns:
            s['num_vars'] *= np.random.uniform(5, 10, len(s))

    def header_mod(s: pd.DataFrame) -> None:
        if 'num_headers' in s.columns:
            s['num_headers'] *= np.random.uniform(8, 15, len(s))
        if 'headers_size_bytes' in s.columns:
            s['headers_size_bytes'] *= np.random.uniform(8, 15, len(s))

    def api_mod(s: pd.DataFrame) -> None:
        if 'method' in s.columns:
            max_v = max(int(s['method'].max()) + 1, 2)
            s['method'] = np.random.randint(0, max_v, len(s))
        if 'path' in s.columns:
            max_v = max(int(s['path'].max()) + 1, 2)
            s['path'] = np.random.randint(0, max_v, len(s))

    def insider_mod(s: pd.DataFrame) -> None:
        for col in ['processing_time_ms', 'vars_size_bytes', 'response_size_bytes']:
            if col in s.columns:
                s[col] *= np.random.uniform(1.3, 2.5, len(s))

    attacks = [
        inject(normal_df, attack_per_type, dos_mod, 'DoS'),
        inject(normal_df, attack_per_type, payload_mod, 'Payload'),
        inject(normal_df, attack_per_type, header_mod, 'HeaderFlood'),
        inject(normal_df, attack_per_type, api_mod, 'APIAbuse'),
        inject(normal_df, attack_per_type, insider_mod, 'InsiderMimic'),
    ]
    full_df = pd.concat([normal_df] + attacks, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return full_df


def build_hg_features(X_in: np.ndarray, feature_names: List[str]) -> np.ndarray:
    df_hg = pd.DataFrame(X_in, columns=feature_names)
    aug_parts = [X_in.astype(np.float32)]
    for col in ['src_ip', 'path', 'method']:
        if col not in df_hg.columns:
            continue
        key = df_hg[col].round(1)
        grp_df = pd.DataFrame({'key': key, 'val': df_hg[col].astype(np.float32)})
        mean_feat = grp_df.groupby('key')['val'].transform('mean').to_numpy(dtype=np.float32)
        std_feat = grp_df.groupby('key')['val'].transform('std').fillna(0.0).to_numpy(dtype=np.float32)
        max_feat = grp_df.groupby('key')['val'].transform('max').to_numpy(dtype=np.float32)
        aug_parts.extend([mean_feat[:, None], std_feat[:, None], max_feat[:, None]])
    return np.hstack(aug_parts).astype(np.float32)


def prepare_data(csv_path: str, scaler_path: str) -> PreparedData:
    df = pd.read_csv(csv_path)
    df = sanitize_columns(df)
    df = encode_dataframe(df)
    full_df = inject_attacks_with_types(df)

    all_features = [c for c in full_df.columns if c not in ['label', 'attack_type']]
    X = full_df[all_features].values.astype(np.float32)
    y = full_df['label'].values.astype(np.int32)
    attack = full_df['attack_type'].astype(str).values

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X).astype(np.float32)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        joblib.dump(scaler, scaler_path)

    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, attack, test_size=0.20, random_state=42, stratify=attack
    )
    X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
        X_train, y_train, a_train, test_size=0.15, random_state=42, stratify=a_train
    )

    return PreparedData(
        all_features=all_features,
        X_train_raw=X_train,
        X_val_raw=X_val,
        X_test_raw=X_test,
        X_train_hg=build_hg_features(X_train, all_features),
        X_val_hg=build_hg_features(X_val, all_features),
        X_test_hg=build_hg_features(X_test, all_features),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        attack_train=a_train,
        attack_val=a_val,
        attack_test=a_test,
        scaler=scaler,
    )


def load_raven_model(model_path: str) -> Model:
    return load_model(model_path, custom_objects={'VAESampling': VAESampling}, compile=False)


def get_encoder(model: Model) -> Model:
    for layer_name in ['z_mean', 'latent_dense']:
        try:
            return Model(model.input, model.get_layer(layer_name).output)
        except Exception:
            pass
    raise ValueError('Could not locate latent layer in model.')


def get_attention_model(model: Model) -> Model:
    try:
        return Model(model.input, model.get_layer('attn_sig').output)
    except Exception:
        return None


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Bal-Acc': balanced_accuracy_score(y_true, y_pred),
    }


def savefig(fig, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + '.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(path_no_ext + '.svg', bbox_inches='tight', facecolor='white')
    fig.savefig(path_no_ext + '.png', dpi=900, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def ensure_dirs(out_root: str) -> Dict[str, str]:
    dirs = {
        'root': out_root,
        'individual': os.path.join(out_root, 'individual'),
        'tables': os.path.join(out_root, 'tables'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def subsample_by_type(X: np.ndarray, y: np.ndarray, attack: np.ndarray, max_per_class: int = 500):
    idx_all = []
    rng = np.random.default_rng(42)
    for c in ATTACK_ORDER:
        idx = np.where(attack == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        idx_all.extend(idx.tolist())
    idx_all = np.array(sorted(idx_all))
    return X[idx_all], y[idx_all], attack[idx_all], idx_all


def make_latent_embeddings(model: Model, X_hg: np.ndarray, y: np.ndarray, attack: np.ndarray):
    encoder = get_encoder(model)
    Xs, ys, attacks, _ = subsample_by_type(X_hg, y, attack, max_per_class=380)
    Z = encoder.predict(Xs, verbose=0)
    tsne = TSNE(n_components=2, perplexity=32, init='pca', learning_rate='auto', random_state=42)
    Z_tsne = tsne.fit_transform(Z)
    return Xs, ys, attacks, Z, Z_tsne


def plot_latent_tsne_by_attack(Z_tsne: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name in ATTACK_ORDER:
        m = attack == name
        if m.sum() == 0:
            continue
        ax.scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=6, alpha=0.75, c=ATTACK_COLORS[name], label=name, rasterized=True)
    apply_axis_style(ax, 't-SNE 1', 't-SNE 2')
    ax.legend(ncol=2, loc='best', fontsize=5.6, handletextpad=0.3, columnspacing=0.7, markerscale=1.2)
    savefig(fig, save_path)


def plot_hg_vs_raw_pca(X_raw: np.ndarray, X_hg: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    pca_raw = PCA(n_components=2, random_state=42).fit_transform(X_raw)
    pca_hg = PCA(n_components=2, random_state=42).fit_transform(X_hg)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)

    for ax, data, title in zip(axes, [pca_raw, pca_hg], ['Raw', 'HyperGraph']):
        for name in ATTACK_ORDER:
            m = attack == name
            if m.sum() == 0:
                continue
            ax.scatter(data[m, 0], data[m, 1], s=5, alpha=0.70, c=ATTACK_COLORS[name], rasterized=True)
        apply_axis_style(ax, 'PC1', 'PC2')
        ax.set_title(title, fontsize=7.5, pad=2)

    handles = [
        Line2D([0], [0], marker='o', color='w', label=n,
               markerfacecolor=ATTACK_COLORS[n], markersize=4.5)
        for n in ATTACK_ORDER
    ]
    axes[1].legend(handles=handles, loc='best', fontsize=5.0, handletextpad=0.25, borderpad=0.1)
    savefig(fig, save_path)


def plot_latent_correlation_heatmap(Z: np.ndarray, save_path: str) -> None:
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    apply_axis_style(ax, 'Latent Dim', 'Latent Dim')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    savefig(fig, save_path)


def plot_attack_centroid_distance(Z: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    centroids = []
    labels = []
    for name in ATTACK_ORDER:
        m = attack == name
        if m.sum() == 0:
            continue
        centroids.append(Z[m].mean(axis=0))
        labels.append(name)
    centroids = np.stack(centroids, axis=0)
    dmat = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(dmat, cmap='magma')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=5.5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = dmat[i, j]
            ax.text(
                j, i, f'{val:.2f}',
                ha='center', va='center',
                color='white' if val > dmat.max() * 0.55 else 'black',
                fontsize=5.2
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6, length=2)
    savefig(fig, save_path)


def plot_attention_heatmap_by_type(att: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    order_idx = np.argsort(pd.Categorical(attack, categories=ATTACK_ORDER, ordered=True).codes)
    att_sorted = att[order_idx]
    attack_sorted = attack[order_idx]

    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    im = ax.imshow(att_sorted.T, aspect='auto', interpolation='nearest', cmap='viridis')
    apply_axis_style(ax, 'Samples', 'Channels')

    start = 0
    ticks = []
    ticklabels = []
    for name in ATTACK_ORDER:
        count = int((attack_sorted == name).sum())
        if count == 0:
            continue
        mid = start + count / 2.0
        ticks.append(mid)
        ticklabels.append(name)
        ax.axvline(start, color='white', linewidth=0.6, alpha=0.8)
        start += count

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=25, fontsize=5.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.4, length=2)
    savefig(fig, save_path)


def plot_attention_signature_by_type(att: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    for name in ATTACK_ORDER:
        m = attack == name
        if m.sum() == 0:
            continue
        ax.plot(att[m].mean(axis=0), label=name, color=ATTACK_COLORS[name], linewidth=0.9)
    apply_axis_style(ax, 'Channel', 'Mean Weight')
    ax.legend(ncol=2, fontsize=4.9, loc='best', handlelength=1.2, columnspacing=0.6)
    savefig(fig, save_path)


def plot_score_distributions_by_attack(y_prob: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    bins = np.linspace(0, 1, 36)
    for name in ATTACK_ORDER:
        m = attack == name
        if m.sum() == 0:
            continue
        ax.hist(
            y_prob[m],
            bins=bins,
            density=True,
            alpha=0.30,
            color=ATTACK_COLORS[name],
            label=name,
            linewidth=0
        )
    ax.axvline(0.5, linestyle='--', linewidth=0.8, color='gray')
    apply_axis_style(ax, 'Attack Probability', 'Density')
    ax.legend(ncol=2, fontsize=4.9, loc='best', handlelength=1.0, columnspacing=0.5)
    savefig(fig, save_path)


def plot_attackwise_roc(y_true: np.ndarray, y_prob: np.ndarray, attack: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.plot([0, 1], [0, 1], '--', linewidth=0.7, color='gray')
    for name in ATTACK_ORDER[1:]:
        mask = np.logical_or(attack == 'Normal', attack == name)
        y_bin = (attack[mask] == name).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        prob = y_prob[mask]
        fpr, tpr, _ = roc_curve(y_bin, prob)
        auc = roc_auc_score(y_bin, prob)
        ax.plot(fpr, tpr, label=f'{name} ({auc:.3f})', color=ATTACK_COLORS[name], linewidth=0.95)
    apply_axis_style(ax, 'FPR', 'TPR')
    ax.legend(fontsize=4.9, loc='lower right', handlelength=1.1, borderpad=0.2)
    savefig(fig, save_path)


def main():
    parser = argparse.ArgumentParser(description='Compact IEEE-ready RAVEN-ADE++ visual suite')
    parser.add_argument('--csv-path', required=True, help='Path to keystone_features_parsed_struct.csv')
    parser.add_argument('--model-path', required=True, help='Path to raven_ade_full.keras')
    parser.add_argument('--scaler-path', required=True, help='Path to scaler.pkl')
    parser.add_argument('--out-dir', default='raven_modulewise_outputs_compact', help='Output directory')
    args = parser.parse_args()

    set_pub_style()
    t0 = time.time()
    dirs = ensure_dirs(args.out_dir)

    data = prepare_data(args.csv_path, args.scaler_path)
    model = load_raven_model(args.model_path)

    y_prob = model.predict(data.X_test_hg, verbose=0).ravel()
    metrics = evaluate(data.y_test, y_prob)

    Xs_raw, _, atk_raw, _ = subsample_by_type(data.X_test_raw, data.y_test, data.attack_test, max_per_class=380)
    Xs_hg, _, atk_hg, _ = subsample_by_type(data.X_test_hg, data.y_test, data.attack_test, max_per_class=380)
    _, _, atk_embed, Z, Z_tsne = make_latent_embeddings(model, data.X_test_hg, data.y_test, data.attack_test)

    att_model = get_attention_model(model)
    if att_model is None:
        raise RuntimeError('Attention layer attn_sig not found in the loaded model.')
    Xs_att, _, atk_att, _ = subsample_by_type(data.X_test_hg, data.y_test, data.attack_test, max_per_class=380)
    att = att_model.predict(Xs_att, verbose=0)

    module_paths = {
        'latent_by_attack': os.path.join(dirs['individual'], 'module_01_latent_tsne_by_attack'),
        'hg_vs_raw': os.path.join(dirs['individual'], 'module_02_hg_vs_raw_pca'),
        'latent_corr': os.path.join(dirs['individual'], 'module_03_latent_correlation_heatmap'),
        'centroid_dist': os.path.join(dirs['individual'], 'module_04_attack_centroid_distance'),
        'attention_heatmap': os.path.join(dirs['individual'], 'module_05_attention_heatmap_by_type'),
        'attention_signature': os.path.join(dirs['individual'], 'module_06_attention_signature_by_type'),
        'score_dist': os.path.join(dirs['individual'], 'module_07_score_distribution_by_attack'),
        'roc_attackwise': os.path.join(dirs['individual'], 'module_08_attackwise_roc'),
    }

    plot_latent_tsne_by_attack(Z_tsne, atk_embed, module_paths['latent_by_attack'])
    plot_hg_vs_raw_pca(Xs_raw, Xs_hg, atk_hg, module_paths['hg_vs_raw'])
    plot_latent_correlation_heatmap(Z, module_paths['latent_corr'])
    plot_attack_centroid_distance(Z, atk_embed, module_paths['centroid_dist'])
    plot_attention_heatmap_by_type(att, atk_att, module_paths['attention_heatmap'])
    plot_attention_signature_by_type(att, atk_att, module_paths['attention_signature'])

    X_prob, _, atk_prob, _ = subsample_by_type(data.X_test_hg, data.y_test, data.attack_test, max_per_class=380)
    y_prob_sub = model.predict(X_prob, verbose=0).ravel()
    plot_score_distributions_by_attack(y_prob_sub, atk_prob, module_paths['score_dist'])
    plot_attackwise_roc(data.y_test, y_prob, data.attack_test, module_paths['roc_attackwise'])

    stats = {
        'metrics': metrics,
        'dataset': {
            'Total Samples': int(len(data.y_train) + len(data.y_val) + len(data.y_test)),
            'Train Split': int(len(data.y_train)),
            'Val Split': int(len(data.y_val)),
            'Test Split': int(len(data.y_test)),
            'Raw Features': int(data.X_train_raw.shape[1]),
            'HG-Aug Features': int(data.X_train_hg.shape[1]),
        },
        'runtime_seconds': round(time.time() - t0, 2),
    }

    with open(os.path.join(dirs['tables'], 'modulewise_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('Done.')
    print(f'Output directory: {dirs["root"]}')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()