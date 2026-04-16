import os
import sys
import time
import json
import warnings
import logging
import argparse

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Layer,
    Multiply,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = r"D:\Raven"
CSV_PATH = os.path.join(BASE_DIR, "data", "keystone_features_parsed_struct.csv")
OUT_ROOT = os.path.join(BASE_DIR, "outputs")
OUT_TABLES = os.path.join(OUT_ROOT, "tables")
OUT_MODELS = os.path.join(OUT_ROOT, "models")
OUT_HD = os.path.join(OUT_ROOT, "figures_hd_ieee")
OUT_LOGS = os.path.join(OUT_ROOT, "logs")

for d in [OUT_TABLES, OUT_MODELS, OUT_HD, OUT_LOGS]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_LOGS, "raven_visual_ieee_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("raven_visual_ieee")

LATENT_DIM = 32
HIDDEN_DIMS = [256, 128, 64]
BETA = 1.0


def set_pub_style():
    plt.rcParams.update({
        'figure.dpi': 180,
        'savefig.dpi': 420,
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.20,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'lines.linewidth': 1.9,
    })


class VAESampling(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        kl_loss = -0.5 * tf.reduce_mean(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        self.add_loss(self.beta * tf.reduce_mean(kl_loss))
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'beta': self.beta})
        return cfg


def build_raven(input_dim, use_vae=True, use_attention=True):
    inp = Input(shape=(input_dim,), name='input_features')
    x = inp
    for i, u in enumerate(HIDDEN_DIMS):
        x = Dense(u, activation='elu', kernel_regularizer=l2(1e-4), name=f'enc_dense_{i+1}')(x)
        x = BatchNormalization(name=f'enc_bn_{i+1}')(x)
        x = Dropout(0.2, name=f'enc_do_{i+1}')(x)

    if use_attention:
        a = Dense(HIDDEN_DIMS[-1], activation='tanh', name='attn_pre')(x)
        a = Dense(HIDDEN_DIMS[-1], activation='sigmoid', name='attn_sig')(a)
        x = Multiply(name='attn_mul')([x, a])

    if use_vae:
        z_mean = Dense(LATENT_DIM, name='z_mean')(x)
        z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)
        enc_out = VAESampling(beta=BETA, name='z')([z_mean, z_log_var])
    else:
        enc_out = Dense(LATENT_DIM, activation='elu', name='latent_dense')(x)

    h = Dense(64, activation='elu', name='head_dense')(enc_out)
    h = Dropout(0.3, name='head_do')(h)
    out = Dense(1, activation='sigmoid', name='prob_attack')(h)
    m = Model(inp, out, name='raven_ade')
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return m


def evaluate(y_true, y_prob, thr=0.5):
    yp = (y_prob >= thr).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, yp),
        'Precision': precision_score(y_true, yp, zero_division=0),
        'Recall': recall_score(y_true, yp, zero_division=0),
        'F1': f1_score(y_true, yp, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, yp),
        'Bal-Acc': balanced_accuracy_score(y_true, yp),
    }


def sanitize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.drop_duplicates().reset_index(drop=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['dayofweek'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
        df = df.drop(columns=['timestamp'])
    return df


def inject_attacks(df):
    normal_df = df.copy()
    normal_df['label'] = 0
    n_normal = len(normal_df)
    attack_per_type = n_normal // 5

    def inject(base_df, n, mod_fn, name):
        s = base_df.sample(n=min(n, len(base_df)), replace=True, random_state=42).copy()
        mod_fn(s)
        s['label'] = 1
        s['attack_type'] = name
        return s

    def dos_mod(s):
        if 'processing_time_ms' in s:
            s['processing_time_ms'] *= np.random.uniform(8, 15, len(s))

    def payload_mod(s):
        if 'vars_size_bytes' in s:
            s['vars_size_bytes'] *= np.random.uniform(10, 20, len(s))
        if 'num_vars' in s:
            s['num_vars'] *= np.random.uniform(5, 10, len(s))

    def header_mod(s):
        if 'num_headers' in s:
            s['num_headers'] *= np.random.uniform(8, 15, len(s))
        if 'headers_size_bytes' in s:
            s['headers_size_bytes'] *= np.random.uniform(8, 15, len(s))

    def api_mod(s):
        if 'method' in s:
            s['method'] = np.random.randint(0, max(int(s['method'].max()) + 1, 2), len(s))
        if 'path' in s:
            s['path'] = np.random.randint(0, max(int(s['path'].max()) + 1, 2), len(s))

    def insider_mod(s):
        for col in ['processing_time_ms', 'vars_size_bytes', 'response_size_bytes']:
            if col in s:
                s[col] *= np.random.uniform(1.3, 2.5, len(s))

    attacks = [
        inject(normal_df, attack_per_type, dos_mod, 'DoS'),
        inject(normal_df, attack_per_type, payload_mod, 'Payload'),
        inject(normal_df, attack_per_type, header_mod, 'HeaderFlood'),
        inject(normal_df, attack_per_type, api_mod, 'APIAbuse'),
        inject(normal_df, attack_per_type, insider_mod, 'InsiderMimic'),
    ]
    attack_df = pd.concat(attacks, ignore_index=True)
    full_df = pd.concat([normal_df, attack_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return full_df.drop(columns=['attack_type'], errors='ignore')


def build_hg_features(X_in, feature_names):
    if len(feature_names) != X_in.shape[1]:
        feature_names = [f'f{i}' for i in range(X_in.shape[1])]
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


def prepare_data(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = sanitize_columns(df)
    cat_cols = [c for c in ['method', 'path', 'http_version', 'application_number', 'request_sequence', 'src_ip'] if c in df.columns]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    full_df = inject_attacks(df)
    all_features = [c for c in full_df.columns if c != 'label']
    X = full_df[all_features].values.astype(np.float32)
    y = full_df['label'].values.astype(np.float32)

    scaler_path = os.path.join(OUT_MODELS, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X).astype(np.float32)
        log.info(f"[LOAD] scaler.pkl loaded from {scaler_path}")
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        joblib.dump(scaler, scaler_path)
        log.info(f"[SAVE] scaler.pkl saved to {scaler_path}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    return {
        'df': df,
        'full_df': full_df,
        'all_features': all_features,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_hg': build_hg_features(X_train, all_features),
        'X_val_hg': build_hg_features(X_val, all_features),
        'X_test_hg': build_hg_features(X_test, all_features),
    }


def get_callbacks():
    return [EarlyStopping(patience=10, restore_best_weights=True, verbose=0), ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]


def train_or_load_model(data, model_path=None, force_retrain=False):
    if model_path is None:
        model_path = os.path.join(OUT_MODELS, 'raven_ade_full.keras')
    hg_dim = data['X_train_hg'].shape[1]
    if os.path.exists(model_path) and not force_retrain:
        log.info(f"[LOAD] Loading trained model: {model_path}")
        model = load_model(model_path, custom_objects={'VAESampling': VAESampling}, compile=False)
        model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        history = None
    else:
        log.info('[TRAIN] Training RAVEN-ADE++ model...')
        model = build_raven(hg_dim)
        history = model.fit(data['X_train_hg'], data['y_train'], validation_data=(data['X_val_hg'], data['y_val']), epochs=60, batch_size=256, callbacks=get_callbacks(), verbose=0)
        model.save(model_path)
        log.info(f"[SAVE] Model saved to {model_path}")
    return model, history


def encoder_model(model):
    try:
        return Model(model.input, model.get_layer('z_mean').output, name='encoder_mean')
    except Exception:
        return Model(model.input, model.get_layer('latent_dense').output, name='encoder_mean_fallback')


def attention_model(model):
    try:
        return Model(model.input, model.get_layer('attn_sig').output, name='attention_extractor')
    except Exception:
        return None


def ensure_dirs(prefix):
    base = os.path.join(OUT_HD, prefix)
    indiv = os.path.join(base, 'individual')
    panels = os.path.join(base, 'panels')
    os.makedirs(indiv, exist_ok=True)
    os.makedirs(panels, exist_ok=True)
    return base, indiv, panels


def save_figure(fig, path_no_ext):
    png = path_no_ext + '.png'
    pdf = path_no_ext + '.pdf'
    fig.savefig(png, dpi=420, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info(f'[FIG] {png}')


def subsample(X, y, probs=None, max_n=4000):
    n = len(X)
    if n <= max_n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_n, replace=False)
        idx.sort()
    if probs is None:
        return X[idx], y[idx], idx
    return X[idx], y[idx], probs[idx], idx


def make_latent_cache(model, X_hg, y, max_n=4000):
    enc = encoder_model(model)
    Xs, ys, idx = subsample(X_hg, y, max_n=max_n)
    latent = enc.predict(Xs, verbose=0)
    pca2 = PCA(n_components=2, random_state=42)
    lat_pca2 = pca2.fit_transform(latent)
    pca8 = PCA(n_components=min(8, latent.shape[1]), random_state=42)
    lat_pca8 = pca8.fit_transform(latent)
    tsne_idx = np.linspace(0, len(latent) - 1, min(len(latent), 1800), dtype=int)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=35, random_state=42)
    lat_tsne = tsne.fit_transform(latent[tsne_idx])
    mean_n = latent[ys == 0].mean(axis=0)
    mean_a = latent[ys == 1].mean(axis=0)
    var_all = latent.var(axis=0)
    delta_mean = mean_a - mean_n
    snr = np.abs(delta_mean) / (latent.std(axis=0) + 1e-8)
    top_shift = np.argsort(np.abs(delta_mean))[::-1][:10]
    cov = np.corrcoef(latent, rowvar=False)
    if np.isnan(cov).any():
        cov = np.nan_to_num(cov)
    return {
        'latent': latent,
        'ys': ys,
        'lat_pca2': lat_pca2,
        'pca2': pca2,
        'lat_pca8': lat_pca8,
        'pca8': pca8,
        'lat_tsne': lat_tsne,
        'tsne_y': ys[tsne_idx],
        'mean_n': mean_n,
        'mean_a': mean_a,
        'var_all': var_all,
        'delta_mean': delta_mean,
        'snr': snr,
        'top_shift': top_shift,
        'cov': cov,
    }


def save_main_figures(y_true, y_prob, metrics, indiv_dir, panels_dir):
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

    # individual figures
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], '--', linewidth=1)
    ax.set_title(f'ROC Curve (AUC={metrics["ROC-AUC"]:.4f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    save_figure(fig, os.path.join(indiv_dir, '01_roc_curve'))

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot(rec, prec)
    ax.set_title(f'Precision–Recall Curve (AP={metrics["PR-AUC"]:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    save_figure(fig, os.path.join(indiv_dir, '02_pr_curve'))

    fig, ax = plt.subplots(figsize=(6.0, 5.3))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0, 1], ['Normal', 'Attack'])
    ax.set_yticks([0, 1], ['Normal', 'Attack'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, os.path.join(indiv_dir, '03_confusion_matrix'))

    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.72, density=True, label='Normal')
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.72, density=True, label='Attack')
    ax.axvline(0.5, linestyle='--', linewidth=1)
    ax.set_title('Predicted Score Distribution')
    ax.set_xlabel('Attack Probability')
    ax.set_ylabel('Density')
    ax.legend()
    save_figure(fig, os.path.join(indiv_dir, '04_score_distribution'))

    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    ax.plot([0, 1], [0, 1], '--', linewidth=1)
    ax.plot(prob_pred, prob_true, marker='o')
    ax.set_title('Calibration Curve')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Attack Rate')
    save_figure(fig, os.path.join(indiv_dir, '05_calibration_curve'))

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8), constrained_layout=True)
    axes[0].plot(fpr, tpr); axes[0].plot([0,1],[0,1],'--',linewidth=1); axes[0].set_title(f'ROC AUC={metrics["ROC-AUC"]:.4f}'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[1].plot(rec, prec); axes[1].set_title(f'PR AP={metrics["PR-AUC"]:.4f}'); axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    im = axes[2].imshow(cm, interpolation='nearest', cmap='Blues'); axes[2].set_title('Confusion Matrix'); axes[2].set_xticks([0,1],['Normal','Attack']); axes[2].set_yticks([0,1],['Normal','Attack'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].hist(y_prob[y_true==0], bins=40, alpha=0.72, density=True, label='Normal'); axes[3].hist(y_prob[y_true==1], bins=40, alpha=0.72, density=True, label='Attack'); axes[3].axvline(0.5, linestyle='--', linewidth=1); axes[3].set_title('Score Distribution'); axes[3].legend()
    axes[4].plot([0,1],[0,1],'--',linewidth=1); axes[4].plot(prob_pred, prob_true, marker='o'); axes[4].set_title('Calibration')
    fig.suptitle('RAVEN-ADE++ Evaluation Panel (1×5)', fontsize=15, fontweight='bold')
    save_figure(fig, os.path.join(panels_dir, 'panel_main_1x5'))


def save_latent_figures(cache, indiv_dir, panels_dir):
    latent = cache['latent']; ys = cache['ys']; lat_pca2 = cache['lat_pca2']; pca2 = cache['pca2']; lat_tsne = cache['lat_tsne']; tsne_y = cache['tsne_y']
    mean_n = cache['mean_n']; mean_a = cache['mean_a']; var_all = cache['var_all']; delta_mean = cache['delta_mean']; snr = cache['snr']; top_shift = cache['top_shift']; cov = cache['cov']

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.scatter(lat_pca2[ys == 0, 0], lat_pca2[ys == 0, 1], s=10, alpha=0.52, label='Normal')
    ax.scatter(lat_pca2[ys == 1, 0], lat_pca2[ys == 1, 1], s=10, alpha=0.52, label='Attack')
    ax.set_title('Latent Space via PCA')
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(markerscale=2)
    save_figure(fig, os.path.join(indiv_dir, '06_latent_pca'))

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.scatter(lat_tsne[tsne_y == 0, 0], lat_tsne[tsne_y == 0, 1], s=10, alpha=0.52, label='Normal')
    ax.scatter(lat_tsne[tsne_y == 1, 0], lat_tsne[tsne_y == 1, 1], s=10, alpha=0.52, label='Attack')
    ax.set_title('Latent Space via t-SNE')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2)
    save_figure(fig, os.path.join(indiv_dir, '07_latent_tsne'))

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    top_var = np.argsort(var_all)[::-1][:12]
    ax.bar(np.arange(len(top_var)), var_all[top_var])
    ax.set_title('Top Latent Dimensions by Variance')
    ax.set_xlabel('Ranked Latent Dimension')
    ax.set_ylabel('Variance')
    save_figure(fig, os.path.join(indiv_dir, '08_latent_variance_bar'))

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    centroid_gap = np.abs(mean_a[top_shift] - mean_n[top_shift])
    ax.bar(np.arange(len(top_shift)), centroid_gap)
    ax.set_title('Class Centroid Separation in Latent Space')
    ax.set_xlabel('Selected Latent Dimension')
    ax.set_ylabel('|μ_attack − μ_normal|')
    save_figure(fig, os.path.join(indiv_dir, '09_latent_centroid_separation'))

    fig, ax = plt.subplots(figsize=(7.3, 5.3))
    ax.plot(mean_n, label='Normal mean')
    ax.plot(mean_a, label='Attack mean')
    ax.set_title('Latent Mean Profiles')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Activation')
    ax.legend()
    save_figure(fig, os.path.join(indiv_dir, '10_latent_mean_profiles'))

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    im = ax.imshow(cov, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Latent Correlation Heatmap')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Latent Dimension')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, os.path.join(indiv_dir, '11_latent_correlation_heatmap'))

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.bar(np.arange(len(top_shift)), snr[top_shift])
    ax.set_title('Latent Discriminative Strength')
    ax.set_xlabel('Top Shifted Latent Dimensions')
    ax.set_ylabel('|Δμ| / σ')
    save_figure(fig, os.path.join(indiv_dir, '12_latent_discriminative_strength'))

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    xdim, ydim = top_shift[0], top_shift[1] if len(top_shift) > 1 else (0, 1)
    ax.scatter(latent[ys == 0, xdim], latent[ys == 0, ydim], s=10, alpha=0.5, label='Normal')
    ax.scatter(latent[ys == 1, xdim], latent[ys == 1, ydim], s=10, alpha=0.5, label='Attack')
    ax.set_title('Top Latent Pair Separation')
    ax.set_xlabel(f'z[{xdim}]')
    ax.set_ylabel(f'z[{ydim}]')
    ax.legend(markerscale=2)
    save_figure(fig, os.path.join(indiv_dir, '13_latent_top_pair_scatter'))

    fig, axes = plt.subplots(1, 6, figsize=(31, 4.8), constrained_layout=True)
    axes[0].scatter(lat_pca2[ys == 0, 0], lat_pca2[ys == 0, 1], s=9, alpha=0.5); axes[0].scatter(lat_pca2[ys == 1, 0], lat_pca2[ys == 1, 1], s=9, alpha=0.5); axes[0].set_title('PCA')
    axes[1].scatter(lat_tsne[tsne_y == 0, 0], lat_tsne[tsne_y == 0, 1], s=9, alpha=0.5); axes[1].scatter(lat_tsne[tsne_y == 1, 0], lat_tsne[tsne_y == 1, 1], s=9, alpha=0.5); axes[1].set_title('t-SNE')
    tv = np.argsort(var_all)[::-1][:8]; axes[2].bar(np.arange(len(tv)), var_all[tv]); axes[2].set_title('Variance')
    axes[3].bar(np.arange(len(top_shift)), np.abs(delta_mean[top_shift])); axes[3].set_title('Centroid Shift')
    axes[4].plot(mean_n, label='Normal'); axes[4].plot(mean_a, label='Attack'); axes[4].set_title('Mean Profiles'); axes[4].legend()
    im = axes[5].imshow(cov, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1); axes[5].set_title('Correlation')
    fig.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
    fig.suptitle('RAVEN-ADE++ Latent-Space Diagnostics Panel (1×6)', fontsize=15, fontweight='bold')
    save_figure(fig, os.path.join(panels_dir, 'panel_latent_1x6'))


def save_attention_figures(model, X_hg, y, feature_names_hg, indiv_dir, panels_dir):
    att_model = attention_model(model)
    if att_model is None:
        log.info('[SKIP] Attention figures skipped because attn_sig layer is unavailable.')
        return
    Xs, ys, idx = subsample(X_hg, y, max_n=3500)
    att = att_model.predict(Xs, verbose=0)
    mean_n = att[ys == 0].mean(axis=0)
    mean_a = att[ys == 1].mean(axis=0)
    delta = mean_a - mean_n
    top = np.argsort(np.abs(delta))[::-1][:12]
    names = [feature_names_hg[i] if i < len(feature_names_hg) else f'hg_{i-len(feature_names_hg)}' for i in top]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.imshow(att[: min(180, len(att)), :].T, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_title('Attention Heatmap')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Attention Channel')
    save_figure(fig, os.path.join(indiv_dir, '14_attention_heatmap'))

    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    ax.plot(mean_n, label='Normal')
    ax.plot(mean_a, label='Attack')
    ax.set_title('Mean Attention Signature')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Average Weight')
    ax.legend()
    save_figure(fig, os.path.join(indiv_dir, '15_attention_mean_signature'))

    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.barh(np.arange(len(top))[::-1], np.abs(delta[top])[::-1])
    ax.set_yticks(np.arange(len(top))[::-1], names[::-1])
    ax.set_title('Most Shifted Attention Channels')
    ax.set_xlabel('|Attack − Normal|')
    save_figure(fig, os.path.join(indiv_dir, '16_attention_shifted_channels'))

    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    ax.scatter(mean_n[top], mean_a[top], s=36)
    low = min(mean_n[top].min(), mean_a[top].min()) * 0.95
    high = max(mean_n[top].max(), mean_a[top].max()) * 1.05
    ax.plot([low, high], [low, high], '--', linewidth=1)
    for i, n in enumerate(names):
        ax.annotate(n, (mean_n[top][i], mean_a[top][i]), fontsize=8, alpha=0.8)
    ax.set_title('Normal vs Attack Attention')
    ax.set_xlabel('Normal Mean Weight')
    ax.set_ylabel('Attack Mean Weight')
    save_figure(fig, os.path.join(indiv_dir, '17_attention_normal_vs_attack'))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)
    axes[0].imshow(att[: min(180, len(att)), :].T, aspect='auto', interpolation='nearest', cmap='viridis'); axes[0].set_title('Heatmap')
    axes[1].plot(mean_n, label='Normal'); axes[1].plot(mean_a, label='Attack'); axes[1].set_title('Mean Signature'); axes[1].legend()
    axes[2].barh(np.arange(len(top))[::-1], np.abs(delta[top])[::-1]); axes[2].set_yticks(np.arange(len(top))[::-1], names[::-1]); axes[2].set_title('Shifted Channels')
    axes[3].scatter(mean_n[top], mean_a[top], s=30); axes[3].plot([low,high],[low,high],'--',linewidth=1); axes[3].set_title('Normal vs Attack')
    fig.suptitle('RAVEN-ADE++ Attention Diagnostics Panel (1×4)', fontsize=15, fontweight='bold')
    save_figure(fig, os.path.join(panels_dir, 'panel_attention_1x4'))


def save_feature_space_figures(X_hg, y, y_prob, feature_names_hg, indiv_dir, panels_dir):
    Xs, ys, probs, idx = subsample(X_hg, y, y_prob, max_n=4000)
    pca = PCA(n_components=2, random_state=42)
    p2 = pca.fit_transform(Xs)
    mean_attack = Xs[ys == 1].mean(axis=0)
    mean_normal = Xs[ys == 0].mean(axis=0)
    delta = np.abs(mean_attack - mean_normal)
    top = np.argsort(delta)[::-1][:12]
    names = [feature_names_hg[i] if i < len(feature_names_hg) else f'hg_{i-len(feature_names_hg)}' for i in top]

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    sc = ax.scatter(p2[:, 0], p2[:, 1], c=probs, s=11, alpha=0.62, cmap='viridis')
    ax.set_title('HG Feature Space via PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Attack Prob.')
    save_figure(fig, os.path.join(indiv_dir, '18_hg_pca_space'))

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(np.arange(len(top))[::-1], delta[top][::-1])
    ax.set_yticks(np.arange(len(top))[::-1], names[::-1])
    ax.set_title('Top Discriminative HG Features')
    ax.set_xlabel('|μ_attack − μ_normal|')
    save_figure(fig, os.path.join(indiv_dir, '19_hg_top_discriminative_features'))

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(Xs[:, top[0]], Xs[:, top[1]], c=ys, s=11, alpha=0.6)
    ax.set_title('Top HG Feature Pair Separation')
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    save_figure(fig, os.path.join(indiv_dir, '20_hg_top_pair_scatter'))

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ranks = np.argsort(probs)
    ax.plot(probs[ranks])
    ax.axhline(0.5, linestyle='--', linewidth=1)
    ax.set_title('Sorted Attack Scores')
    ax.set_xlabel('Sorted Sample Index')
    ax.set_ylabel('Predicted Probability')
    save_figure(fig, os.path.join(indiv_dir, '21_sorted_attack_scores'))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)
    sc = axes[0].scatter(p2[:,0], p2[:,1], c=probs, s=10, alpha=0.62, cmap='viridis'); axes[0].set_title('HG PCA'); fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].barh(np.arange(len(top))[::-1], delta[top][::-1]); axes[1].set_yticks(np.arange(len(top))[::-1], names[::-1]); axes[1].set_title('HG Features')
    axes[2].scatter(Xs[:, top[0]], Xs[:, top[1]], c=ys, s=10, alpha=0.6); axes[2].set_title('Feature Pair')
    axes[3].plot(probs[ranks]); axes[3].axhline(0.5, linestyle='--', linewidth=1); axes[3].set_title('Sorted Scores')
    fig.suptitle('RAVEN-ADE++ Hypergraph Feature Diagnostics Panel (1×4)', fontsize=15, fontweight='bold')
    save_figure(fig, os.path.join(panels_dir, 'panel_hgspace_1x4'))


def save_training_figures(history, indiv_dir, panels_dir):
    if history is None:
        log.info('[SKIP] Training figures skipped because the model was loaded, not trained in this run.')
        return
    h = history.history
    epochs = np.arange(1, len(h['loss']) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(epochs, h['loss'], label='Train')
    ax.plot(epochs, h['val_loss'], label='Val')
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Cross-Entropy')
    ax.legend()
    save_figure(fig, os.path.join(indiv_dir, '22_training_loss_curves'))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(epochs, h.get('accuracy', []), label='Train')
    ax.plot(epochs, h.get('val_accuracy', []), label='Val')
    ax.set_title('Training Accuracy Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_figure(fig, os.path.join(indiv_dir, '23_training_accuracy_curves'))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if len(epochs) > 1:
        dloss = np.diff(h['val_loss'])
        ax.plot(epochs[1:], dloss)
        ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title('Validation Loss Delta')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Δ Val Loss')
    save_figure(fig, os.path.join(indiv_dir, '24_validation_loss_delta'))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    best_epoch = int(np.argmin(h['val_loss'])) + 1
    ax.bar(['Best epoch', 'Total epochs'], [best_epoch, len(epochs)])
    ax.set_title('Training Summary')
    ax.set_ylabel('Epoch Count')
    save_figure(fig, os.path.join(indiv_dir, '25_training_summary'))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), constrained_layout=True)
    axes[0].plot(epochs, h['loss'], label='Train'); axes[0].plot(epochs, h['val_loss'], label='Val'); axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(epochs, h.get('accuracy', []), label='Train'); axes[1].plot(epochs, h.get('val_accuracy', []), label='Val'); axes[1].set_title('Accuracy'); axes[1].legend()
    if len(epochs) > 1: axes[2].plot(epochs[1:], np.diff(h['val_loss'])); axes[2].axhline(0, linestyle='--', linewidth=1)
    axes[2].set_title('Val Loss Delta')
    axes[3].bar(['Best', 'Total'], [best_epoch, len(epochs)]); axes[3].set_title('Summary')
    fig.suptitle('RAVEN-ADE++ Training Diagnostics Panel (1×4)', fontsize=15, fontweight='bold')
    save_figure(fig, os.path.join(panels_dir, 'panel_training_1x4'))


def save_metrics_table(metrics, prefix):
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    metrics_path = os.path.join(OUT_TABLES, f'{prefix}_metrics_summary.csv')
    metrics_df.to_csv(metrics_path)
    with open(os.path.join(OUT_TABLES, f'{prefix}_metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    log.info(f'[SAVED] {metrics_path}')


def generate_visual_suite(model, history, data, prefix='raven_ade_full'):
    _, indiv_dir, panels_dir = ensure_dirs(prefix)
    y_prob = model.predict(data['X_test_hg'], verbose=0).ravel()
    metrics = evaluate(data['y_test'], y_prob)
    save_metrics_table(metrics, prefix)

    feature_names_hg = list(data['all_features']) + [f'hg_{i}' for i in range(data['X_test_hg'].shape[1] - len(data['all_features']))]
    latent_cache = make_latent_cache(model, data['X_test_hg'], data['y_test'], max_n=4500)

    save_main_figures(data['y_test'], y_prob, metrics, indiv_dir, panels_dir)
    save_latent_figures(latent_cache, indiv_dir, panels_dir)
    save_attention_figures(model, data['X_test_hg'], data['y_test'], feature_names_hg, indiv_dir, panels_dir)
    save_feature_space_figures(data['X_test_hg'], data['y_test'], y_prob, feature_names_hg, indiv_dir, panels_dir)
    save_training_figures(history, indiv_dir, panels_dir)

    report = classification_report(data['y_test'], (y_prob >= 0.5).astype(int), target_names=['Normal', 'Attack'])
    with open(os.path.join(OUT_LOGS, f'{prefix}_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    log.info('\n' + report)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='RAVEN-ADE++ IEEE-style visual analytics suite')
    parser.add_argument('--mode', choices=['visualize', 'train_and_visualize'], default='visualize')
    parser.add_argument('--csv-path', default=CSV_PATH)
    parser.add_argument('--model-path', default=os.path.join(OUT_MODELS, 'raven_ade_full.keras'))
    parser.add_argument('--prefix', default='raven_ade_full_ieee')
    parser.add_argument('--force-retrain', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    set_pub_style()
    t0 = time.time()
    log.info('=' * 92)
    log.info('RAVEN-ADE++ — IEEE-Style Individual + Panel Figure Generator')
    log.info('=' * 92)
    log.info(f'[DATA] CSV: {args.csv_path}')
    log.info(f'[MODEL] Path: {args.model_path}')
    log.info(f'[MODE] {args.mode}')

    data = prepare_data(args.csv_path)
    model, history = train_or_load_model(data, model_path=args.model_path, force_retrain=args.force_retrain or args.mode == 'train_and_visualize')
    metrics = generate_visual_suite(model, history, data, prefix=args.prefix)

    log.info(f"[DONE] Total runtime: {time.time() - t0:.2f}s")
    log.info(f"[OUTPUT] Figures root: {os.path.join(OUT_HD, args.prefix)}")
    log.info(f"[FINAL] " + ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()]))


if __name__ == '__main__':
    main()