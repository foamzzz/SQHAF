import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
import scipy.signal

# =========================================================
# Utils
# =========================================================
def _design_fb_butter(srate: float, sub_bands: int, order: int = 4):
    nyq = srate / 2.0
    passband = [8, 18, 26, 34, 42, 50, 58, 66, 74, 82]
    max_bands = min(sub_bands, len(passband))
    fb_ba = []
    for i in range(max_bands):
        low = max(0.1, passband[i])
        high = min(90.0, nyq - 0.1)
        if low >= high:
            continue
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        fb_ba.append((b, a))
    return fb_ba

def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        m = min(a.size, b.size)
        a = a[:m]
        b = b[:m]
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sign_square(r: float) -> float:
    # 对应论文 Sign(r)=sign(r)*r^2 (Eq.26)
    return float(np.sign(r) * (r ** 2))


def _cov(X: np.ndarray) -> np.ndarray:
    # X: (n_ch, n_samp)
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc @ Xc.T / max(1, X.shape[1] - 1)


def _matrix_inv_sqrt(C: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    # C^{-1/2} for EA
    C = 0.5 * (C + C.T)
    w, V = eigh(C)
    w = np.clip(w, reg, None)
    return (V / np.sqrt(w)) @ V.T


def _matrix_sqrt(C: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    C = 0.5 * (C + C.T)
    w, V = eigh(C)
    w = np.clip(w, reg, None)
    return (V * np.sqrt(w)) @ V.T


def _bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    # data: (..., n_samples)
    nyq = fs / 2.0
    low = max(0.1, low)
    high = min(nyq - 0.1, high)
    if low >= high:
        return data.copy()
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def _make_ref(freq: float, fs: float, n_samples: int, n_harmonics: int = 5) -> np.ndarray:
    # Eq.(2), shape: (2K, Ns)
    t = np.arange(n_samples) / fs
    Y = []
    for k in range(1, n_harmonics + 1):
        Y.append(np.sin(2 * np.pi * k * freq * t))
        Y.append(np.cos(2 * np.pi * k * freq * t))
    return np.asarray(Y)


def _trca_train_one_class(X_cls: np.ndarray, reg: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_cls: (n_trials, n_ch, n_samples) for one class
    返回:
      w: (n_ch,)
      z: (n_samples,) = w^T mean_trial
    对应 Eq.(6)-(10)
    """
    n_trials, n_ch, _ = X_cls.shape
    S = np.zeros((n_ch, n_ch), dtype=float)
    Q = np.zeros((n_ch, n_ch), dtype=float)

    for i in range(n_trials):
        Xi = X_cls[i]
        Q += _cov(Xi)
        for j in range(n_trials):
            if i == j:
                continue
            Xj = X_cls[j]
            Xi_c = Xi - Xi.mean(axis=1, keepdims=True)
            Xj_c = Xj - Xj.mean(axis=1, keepdims=True)
            S += Xi_c @ Xj_c.T / max(1, Xi.shape[1] - 1)

    Q = Q + reg * np.eye(n_ch)
    # maximize w^T S w / w^T Q w
    wvals, wvecs = eigh(0.5 * (S + S.T), 0.5 * (Q + Q.T))
    w = wvecs[:, np.argmax(wvals)]
    X_mean = X_cls.mean(axis=0)
    z = w.T @ X_mean
    return w.real, z.real


def _cca_maxcorr(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    """
    X: (n_ch, n_samples), Y: (n_ref_ch, n_samples)
    使用 QR + SVD 计算第一典型相关系数
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("CCA inputs must be 2D matrices with shape (n_features, n_samples).")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"CCA input lengths do not match: {X.shape} vs {Y.shape}.")

    try:
        # reduced QR
        Qx, _ = np.linalg.qr(X.T, mode="reduced")
        Qy, _ = np.linalg.qr(Y.T, mode="reduced")

        # 第一典型相关 = 最大奇异值
        s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
        rho = s[0] if s.size > 0 else 0.0
        return float(rho)
    except np.linalg.LinAlgError:
        return 0.0


# =========================================================
# Main class
# =========================================================
@dataclass
class SUTLSSVEP:
    srate: float
    freqs: List[float]
    n_harmonics: int = 3
    top_m1: int = 4
    n_bands: int = 5
    a: float = 1.25     # FBCCA weight: k^(-a)+b
    b: float = 0.25
    n_jobs: int = 1
    reg: float = 1e-6

    def __post_init__(self):
        self.freqs = list(self.freqs)
        self.n_classes = len(self.freqs)
        self._fitted = False

        # 预生成filterbank（避免每次trial重复设计）
        self.fb_ba_ = _design_fb_butter(self.srate, self.n_bands)

    # ---------- public ----------
    def fit(self, source_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        source_data[sid] = (X, y)
        X: (n_trials, n_ch, n_samples), y: int labels (建议0..Nf-1)
        """
        self.source_ids_ = sorted(list(source_data.keys()))
        self.source_data_raw_ = source_data

        # 0) label map
        self._prepare_label_mapping(source_data)

        # 1) STE: 计算每个源被试可迁移性 ST_m (Eq.17)
        st_scores = self._compute_ste_scores(source_data)

        # 2) 选 top_m1 源被试 + 贡献权重 rho_m (Eq.22)
        ranked = sorted(st_scores.items(), key=lambda kv: kv[1], reverse=True)
        self.selected_ids_ = [sid for sid, _ in ranked[:min(self.top_m1, len(ranked))]]

        vals = np.array([max(0.0, st_scores[s]) for s in self.selected_ids_], dtype=float)
        if vals.sum() < 1e-12:
            vals = np.ones_like(vals)
        self.rho_ = vals / vals.sum()
        self.rho_map_ = {sid: r for sid, r in zip(self.selected_ids_, self.rho_)}

        # 3) DA(EA): 每个被试构建参考协方差，训练时先对齐源数据 (Eq.20-21)
        self.src_ref_cov_ = {}
        self.src_align_mat_ = {}
        self.src_aligned_ = {}
        for sid in self.selected_ids_:
            X, y = source_data[sid]
            Cbar = np.mean([_cov(tr) for tr in X], axis=0)
            self.src_ref_cov_[sid] = Cbar
            A = _matrix_inv_sqrt(Cbar, reg=self.reg)  # C^{-1/2}
            self.src_align_mat_[sid] = A
            X_al = np.asarray([A @ tr for tr in X])
            self.src_aligned_[sid] = (X_al, self._map_y(y))

        # 4) 学 subject-specific TRCA 知识 wm,n / Zm,n
        self.ss_w_ = {sid: {} for sid in self.selected_ids_}
        self.ss_z_ = {sid: {} for sid in self.selected_ids_}
        for sid in self.selected_ids_:
            X_al, y_al = self.src_aligned_[sid]
            for c in range(self.n_classes):
                Xc = X_al[y_al == c]
                if len(Xc) < 2:
                    # fallback
                    w = np.ones(X_al.shape[1]) / np.sqrt(X_al.shape[1])
                    z = w.T @ X_al.mean(axis=0)
                else:
                    w, z = _trca_train_one_class(Xc, reg=self.reg)
                self.ss_w_[sid][c] = w
                self.ss_z_[sid][c] = z

        # 5) 学 generalization 知识 vn / Tn (Eq.23-24 风格)
        self.gen_v_ = {}
        self.gen_t_ = {}
        for c in range(self.n_classes):
            X_means = []
            for sid in self.selected_ids_:
                X_al, y_al = self.src_aligned_[sid]
                Xc = X_al[y_al == c]
                if len(Xc) == 0:
                    continue
                X_means.append(Xc.mean(axis=0))  # (ch, samples)

            if len(X_means) < 2:
                # fallback
                n_ch = next(iter(self.src_aligned_.values()))[0].shape[1]
                v = np.ones(n_ch) / np.sqrt(n_ch)
                t = np.mean(X_means, axis=0) if len(X_means) else np.zeros((n_ch, next(iter(self.src_aligned_.values()))[0].shape[2]))
                t = v.T @ t
            else:
                n_ch = X_means[0].shape[0]
                S2 = np.zeros((n_ch, n_ch))
                Q2 = np.zeros((n_ch, n_ch))
                for i in range(len(X_means)):
                    Xi = X_means[i]
                    Q2 += _cov(Xi)
                    for j in range(len(X_means)):
                        if i == j:
                            continue
                        Xj = X_means[j]
                        Xi_c = Xi - Xi.mean(axis=1, keepdims=True)
                        Xj_c = Xj - Xj.mean(axis=1, keepdims=True)
                        S2 += Xi_c @ Xj_c.T / max(1, Xi.shape[1] - 1)
                Q2 += self.reg * np.eye(n_ch)
                vals, vecs = eigh(0.5 * (S2 + S2.T), 0.5 * (Q2 + Q2.T))
                v = vecs[:, np.argmax(vals)].real
                t = v.T @ np.mean(np.stack(X_means, axis=0), axis=0)
            self.gen_v_[c] = v
            self.gen_t_[c] = t

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_trials, n_ch, n_samples)
        """
        assert self._fitted, "Call fit() first."
        X = np.asarray(X)
        n_trials = X.shape[0]

        # 目标域 EA：用测试集合自身平均协方差构建参考（无监督）
        Cbar_t = np.mean([_cov(tr) for tr in X], axis=0)
        A_t = _matrix_inv_sqrt(Cbar_t, reg=self.reg)
        X_al = np.asarray([A_t @ tr for tr in X])

        y_pred = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_one)(X_al[i]) for i in range(n_trials)
        )
        return np.asarray(y_pred, dtype=int)

    # ---------- internals ----------
    def _prepare_label_mapping(self, source_data):
        ys = []
        for sid, (_, y) in source_data.items():
            ys.extend(np.asarray(y).tolist())
        uniq = sorted(np.unique(ys).tolist())
        # 若标签数量和freq数量一致，直接映射
        if len(uniq) != self.n_classes:
            # 尽量兼容：按出现顺序截断/扩展
            uniq = uniq[:self.n_classes]
        self.label_to_idx_ = {lb: i for i, lb in enumerate(uniq)}
        self.idx_to_label_ = {i: lb for lb, i in self.label_to_idx_.items()}

    def _map_y(self, y):
        return np.asarray([self.label_to_idx_.get(int(v), 0) for v in y], dtype=int)

    def _fbcca_score(self, trial: np.ndarray, c: int) -> float:
        # Eq.(3)(4)(5) with Chebyshev filterbank
        n_samples = trial.shape[1]
        Yf = _make_ref(self.freqs[c], self.srate, n_samples, self.n_harmonics)

        rho = 0.0
        for k, (b, a) in enumerate(self.fb_ba_, start=1):
            # trial: (n_ch, n_samples), 在最后一维滤波
            Xk = scipy.signal.filtfilt(b, a, trial, axis=-1)
            r = _cca_maxcorr(Xk, Yf, reg=self.reg)
            wk = (k ** (-self.a)) + self.b
            rho += wk * r

        return _sign_square(rho)


    def _compute_ste_scores(self, source_data):
        """
        极简版 STE（效果差但快）：
        - 只计算类模板之间的相关性
        - 不做 trial 级预测
        - 不计算 FBCCA
        """
        sids = sorted(source_data.keys())
        st = {}

        # 1) 预计算每个被试每个类的平均模板
        templates = {}
        for sid in sids:
            X, y = source_data[sid]
            y_m = self._map_y(y)
            templates[sid] = {}
            for c in range(self.n_classes):
                Xc = X[y_m == c]
                if len(Xc) > 0:
                    templates[sid][c] = Xc.mean(axis=0)  # (n_ch, n_samples)
                else:
                    templates[sid][c] = X.mean(axis=0)

        # 2) STE：源 m 的模板和其他被试模板的相似度
        for m in sids:
            sim_sum = 0.0
            for i in sids:
                if i == m:
                    continue
                # 计算 m 和 i 在所有类上的模板相似度
                for c in range(self.n_classes):
                    tm = templates[m][c].ravel()
                    ti = templates[i][c].ravel()
                    r = _safe_corr(tm, ti)
                    sim_sum += r 
            
            st[m] = float(sim_sum)

        return st


    def _predict_one(self, tr: np.ndarray) -> int:
        # Eq.(25)(27)(28)
        scores = []
        for c in range(self.n_classes):
            # r1: FBCCA
            r1 = self._fbcca_score(tr, c)

            # r2: generalization knowledge
            vg = self.gen_v_[c]
            tg = self.gen_t_[c]
            r2 = _sign_square(_safe_corr(vg.T @ tr, tg))

            # r3: subject-specific weighted fusion
            r3 = 0.0
            for sid in self.selected_ids_:
                w = self.ss_w_[sid][c]
                z = self.ss_z_[sid][c]
                r3 += self.rho_map_[sid] * _sign_square(_safe_corr(w.T @ tr, z))

            scores.append(r1 + r2 + r3)

        c_hat = int(np.argmax(scores))
        return int(self.idx_to_label_.get(c_hat, c_hat))
