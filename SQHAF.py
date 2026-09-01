from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.signal import butter, filtfilt
import warnings
import numpy as np
from scipy import signal
from scipy.linalg import eigh


# 统一数组类型别名，便于类型注解阅读
ArrayLike = np.ndarray


def _ensure_3d(X: ArrayLike) -> ArrayLike:
    """
    确保输入是三维数组，形状应为：
    (n_trials, n_channels, n_samples)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials, n_channels, n_samples), got {X.shape}.")
    return X


def _center_rows(X: ArrayLike) -> ArrayLike:
    """
    对每一行做去均值（按时间维度 axis=1）。
    假设输入形状通常是 (n_channels, n_samples)。
    """
    return X - X.mean(axis=1, keepdims=True)


def _corr_1d(x: ArrayLike, y: ArrayLike, eps: float = 1e-12) -> float:
    """
    计算两个一维向量的皮尔逊相关（等价于标准化内积）。
    当分母过小（近零向量）时返回 0，避免数值不稳定。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den < eps:
        return 0.0
    return float(np.dot(x, y) / den)


def _cov_spd(X: ArrayLike, reg: float = 1e-6) -> ArrayLike:
    """
    估计协方差矩阵，并施加轻微正则使其更接近 SPD（对称正定）。
    输入 X 形状通常为 (n_channels, n_samples)。
    """
    Xc = _center_rows(np.asarray(X, dtype=float))
    n_samples = Xc.shape[1]
    C = (Xc @ Xc.T) / max(n_samples - 1, 1)
    C = 0.5 * (C + C.T)  # 强制对称
    C += reg * np.trace(C) / max(C.shape[0], 1) * np.eye(C.shape[0])  # trace-scaled 正则
    return C


def _trial_covariances(trials: ArrayLike, reg: float = 1e-6) -> ArrayLike:
    """Return regularized trial covariance estimates for an EEG trial batch."""
    trials = _ensure_3d(trials)
    if len(trials) == 0:
        raise ValueError("Cannot estimate covariance from an empty trial set.")
    return np.stack([_cov_spd(trial, reg=reg) for trial in trials], axis=0)


def _mean_covariance(trials: ArrayLike, reg: float = 1e-6) -> ArrayLike:
    """
    计算一组 trial 的平均协方差。
    trials 形状: (n_trials, n_channels, n_samples)
    """
    covs = _trial_covariances(trials, reg=reg)
    C = covs.mean(axis=0)
    C = 0.5 * (C + C.T)
    C += reg * np.trace(C) / max(C.shape[0], 1) * np.eye(C.shape[0])
    return C


def _entropy_effective_rank(C: ArrayLike, eps: float = 1e-12) -> float:
    """Return entropy effective rank of a symmetric covariance matrix."""
    values = np.linalg.eigvalsh(0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T))
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= eps:
        return float("nan")
    probabilities = values / total
    positive = probabilities > eps
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return float(np.exp(entropy))


def _inv_sqrtm_spd(C: ArrayLike, eps: float = 1e-10) -> ArrayLike:
    """
    计算 SPD 矩阵的逆平方根 C^{-1/2}。
    """
    vals, vecs = eigh(0.5 * (C + C.T))
    vals = np.clip(vals, eps, None)
    return (vecs / np.sqrt(vals)) @ vecs.T


def _sqrtm_spd(C: ArrayLike, eps: float = 1e-10) -> ArrayLike:
    """
    计算 SPD 矩阵的平方根 C^{1/2}。
    """
    vals, vecs = eigh(0.5 * (C + C.T))
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _top_generalized_eigenvector(S: ArrayLike, Q: ArrayLike, reg: float = 1e-6) -> ArrayLike:
    """
    求解广义特征值问题 S w = λ Q w 的最大特征值对应特征向量。
    用于 TRCA / 空间滤波器求解。
    """
    S = 0.5 * (S + S.T)
    Q = 0.5 * (Q + Q.T)
    Q = Q + reg * np.trace(Q) / max(Q.shape[0], 1) * np.eye(Q.shape[0])
    vals, vecs = eigh(S, Q)
    w = vecs[:, np.argmax(vals)]
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        # 退化时返回均匀方向
        return np.ones(Q.shape[0]) / np.sqrt(Q.shape[0])
    return w / norm

def _bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    # data: (..., n_samples)
    nyq = fs / 2.0
    if low >= high:
        return data.copy()
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

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

    # 按特征维(行)去均值，然后转置为 (n_samples, n_features)
    X_t = (X - X.mean(axis=1, keepdims=True)).T
    Y_t = (Y - Y.mean(axis=1, keepdims=True)).T

    try:
        # reduced QR
        Qx, _ = np.linalg.qr(X_t, mode="reduced")
        Qy, _ = np.linalg.qr(Y_t, mode="reduced")

        # 第一典型相关 = 最大奇异值
        s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
        rho = s[0] if s.size > 0 else 0.0
        return float(np.clip(rho, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return 0.0


def _cca_corr(X: ArrayLike, Y: ArrayLike, reg: float = 1e-6) -> float:
    """
    用 QR + SVD 计算第一典型相关系数。
    X, Y: (n_features, n_samples)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("CCA inputs must be 2D matrices with shape (n_features, n_samples).")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"CCA input lengths do not match: {X.shape} vs {Y.shape}.")

    # 与你旧代码一致：按“时间点为行”做 QR
    # 原来是 filtdata.T / template.T
    X_t = _center_rows(X).T   # (n_samples, n_features)
    Y_t = _center_rows(Y).T   # (n_samples, n_features)

    # reduced QR
    Qx, _ = np.linalg.qr(X_t, mode="reduced")
    Qy, _ = np.linalg.qr(Y_t, mode="reduced")

    # CCA 主相关：最大奇异值
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    rho = s[0] if s.size > 0 else 0.0
    return float(np.clip(rho, 0.0, 1.0))

def _fbcca_corr(
    X: ArrayLike,
    Y: ArrayLike = None,   
    reg: float = 1e-6,    
    *,
    srate: float,
    n_bands: int = 5,
) -> float:
    """
    FBCCA score（单目标频率）:
    - 返回 sign_square(加权相关和)
    参数
    ----
    X : (n_ch, n_samples)
        单次trial数据
    srate : 采样率
    target_freq : 当前候选刺激频率
    n_bands : 子带数
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (n_channels, n_samples).")

    # -------- 固定参数--------
    a = 1.25          # FBCCA权重参数
    b = 0.25
    # 子带低截止固定为 8*k
    # ---------------------------------------
    rho = 0.0
    for k in range(1, n_bands + 1):
        Xk = _bandpass(X, srate, 6.0 * k, 90, 3)
        r = _cca_maxcorr(Xk, Y, reg=reg)
        wk = (k ** (-a)) + b
        rho += wk * r

    return rho


def _split_half_template_stability(
    X: ArrayLike,
    y: ArrayLike,
    classes: Sequence,
    random_state: int = 42,
    n_repeats: int = 5,
    split_mode: str = "time_ordered",
) -> float:
    """
    Split-half template consistency (STC).

    By default, trials are divided into the first and second acquisition
    blocks within each class. This preserves the temporal ordering of the
    recording and avoids mixing early and late trials in both halves. The
    legacy ``random`` mode is retained for sensitivity analyses.

    Parameters
    ----------
    X : (n_trials, n_channels, n_samples)
    y : (n_trials,)
    classes : 类别列表
    random_state : int
        Seed used only when ``split_mode="random"``.
    n_repeats : int
        Number of repeated splits. The time-ordered split is deterministic,
        so repeated evaluations provide the same estimate.
    split_mode : {"time_ordered", "random"}
        Split strategy. ``time_ordered`` uses the first and second halves in
        the input acquisition order; ``random`` preserves the legacy behavior.

    Returns
    -------
    stc : float
        稳定性分数，越大表示被试内部一致性越好。
    """
    if split_mode not in {"time_ordered", "random"}:
        raise ValueError("split_mode must be 'time_ordered' or 'random'.")

    rng = np.random.default_rng(random_state)
    repeat_scores = []

    for _ in range(max(1, n_repeats)):
        class_scores = []

        for cls in classes:
            idx = np.where(y == cls)[0]
            if idx.size < 2:
                continue

            idx = idx.copy()
            if split_mode == "random":
                rng.shuffle(idx)

            mid = idx.size // 2
            if mid == 0 or mid == idx.size:
                continue

            Xa = X[idx[:mid]].mean(axis=0)
            Xb = X[idx[mid:]].mean(axis=0)
            class_scores.append(_cca_corr(Xa, Xb))

        if len(class_scores) > 0:
            repeat_scores.append(np.mean(class_scores))

    if len(repeat_scores) == 0:
        return 0.0

    return float(np.clip(np.mean(repeat_scores), 1e-4, None))

def _subject_template_similarity(
    X_src: ArrayLike,
    y_src: ArrayLike,
    X_tar: ArrayLike,
    y_tar: ArrayLike,
    classes: Sequence,
) -> float:
    """
    Sim_{m,t}^{tmpl} = (1/N_f) * sum_k Corr(S_{m,k}, S_{t,k})
    - X_src.shape == (n_trials, n_channels, n_samples)
    - X_tar.shape == (n_trials, n_channels, n_samples)
    - y_src / y_tar.shape == (n_trials,)
    """
    X_src = np.asarray(X_src, dtype=float)
    X_tar = np.asarray(X_tar, dtype=float)
    y_src = np.asarray(y_src)
    y_tar = np.asarray(y_tar)

    corrs: List[float] = []
    for cls in classes:
        src_idx = np.where(y_src == cls)[0]
        tar_idx = np.where(y_tar == cls)[0]
        if src_idx.size == 0 or tar_idx.size == 0:
            continue

        S_src = X_src[src_idx].mean(axis=0)
        S_tar = X_tar[tar_idx].mean(axis=0)
        corrs.append(_cca_corr(S_src, S_tar))

    if not corrs:
        return 0.0
    return float(np.mean(corrs))


def _trial_x_trial_sum(trials: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    为 TRCA 风格目标构造统计量：
    - Q: 各 trial 自相关项之和
    - S: trial 间互相关项之和（i<j 成对）
    """
    n_trials, n_channels, _ = trials.shape
    S = np.zeros((n_channels, n_channels), dtype=float)
    Q = np.zeros((n_channels, n_channels), dtype=float)
    centered = np.stack([_center_rows(trial) for trial in trials], axis=0)
    for Xi in centered:
        Q += Xi @ Xi.T
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            S += centered[i] @ centered[j].T + centered[j] @ centered[i].T
    return S, Q


@dataclass
class _BandSubjectModel:
    """
    某个频带下、某个源被试的模型容器。
    """
    subject: Union[int, str]
    covariance: ArrayLike
    align_matrix: Optional[ArrayLike]
    stability: float
    templates_raw: Dict[Union[int, str], ArrayLike]
    templates_aligned: Dict[Union[int, str], ArrayLike]
    w_filters: Dict[Union[int, str], ArrayLike]


class SQHAF:
    """
    源域选择 + 两阶段对齐 + 三分支融合 的跨被试 SSVEP 识别模型。
    """

    def __init__(
        self,
        freqs: Optional[Sequence[float]] = None,
        Yf: Optional[ArrayLike] = None,
        filterbank: Optional[Sequence] = None,
        filterweights: Optional[Sequence[float]] = None,
        n_sources: Optional[int] = None,
        neighbor_radius: int = 1,
        neighbor_decay: float = 0.5,
        neighbor_strength: float = 1.0,
        reg: float = 1e-6,
        random_state: int = 42,
        enable_stage1=True,
        enable_stage2=True,
        enable_branch_r1=True,
        enable_branch_r2=True,
        enable_branch_r3=True,
        enable_harmonic_branch=True,
        source_score_mode="adaptive", # {"adaptive","robust","stability","similarity","similarity_confidence","legacy_multiplicative","random"}
        source_weight_mode="score",   # {"score","uniform"}
        confidence_lambda=0.2,         # bounded STC penalty for similarity_confidence
        fusion_mode="signed_square",  # {"signed_square","plain_sum","abs_sum"}
        target_alignment_mode="calibration",  # {"calibration", "transductive", "none"}
        stc_split_mode="time_ordered",  # {"time_ordered", "random"}
        )-> None:
        # 基础参数
        self.freqs = None if freqs is None else list(freqs)
        self.Yf = None if Yf is None else np.asarray(Yf, dtype=float)
        self.filterbank = filterbank
        self.filterweights = None if filterweights is None else np.asarray(filterweights, dtype=float)
        self.n_sources = n_sources
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_decay = float(neighbor_decay)
        self.neighbor_strength = float(neighbor_strength)
        self.reg = float(reg)
        self.random_state = int(random_state)
        self.srate = 250

        # 开关参数
        self.enable_stage1 = bool(enable_stage1)
        self.enable_stage2 = bool(enable_stage2)
        self.enable_branch_r1 = bool(enable_branch_r1)
        self.enable_branch_r2 = bool(enable_branch_r2)
        self.enable_branch_r3 = bool(enable_branch_r3)
        # Explicit ablation switch for the harmonic-reference branch (r1).
        # ``enable_branch_r1`` is retained for backward compatibility.
        self.enable_harmonic_branch = bool(enable_harmonic_branch)
        self.source_score_mode = source_score_mode
        self.source_weight_mode = source_weight_mode
        self.confidence_lambda = float(confidence_lambda)
        self.fusion_mode = fusion_mode
        self.target_alignment_mode = target_alignment_mode
        self.stc_split_mode = stc_split_mode
        if self.source_score_mode not in {"adaptive", "robust", "stability", "similarity", "similarity_confidence", "legacy_multiplicative", "random"}:
            raise ValueError(f"Unknown source_score_mode={self.source_score_mode}")
        if not 0.0 <= self.confidence_lambda <= 1.0:
            raise ValueError("confidence_lambda must be between 0 and 1.")
        if self.target_alignment_mode not in {"calibration", "transductive", "none"}:
            raise ValueError(f"Unknown target_alignment_mode={self.target_alignment_mode}")
        if self.stc_split_mode not in {"time_ordered", "random"}:
            raise ValueError(f"Unknown stc_split_mode={self.stc_split_mode}")
        if self.neighbor_radius < 0 or self.neighbor_strength < 0 or self.neighbor_decay < 0:
            raise ValueError("Neighbour parameters must be non-negative.")

        # 训练后属性
        self.classes_: Optional[np.ndarray] = None
        self.class_to_index_: Optional[Dict] = None
        self.n_bands_: int = 5
        self.band_weights_: Optional[np.ndarray] = None
        self.selected_subjects_: Optional[np.ndarray] = None
        self.source_weights_: Optional[np.ndarray] = None
        self.source_scores_: Optional[np.ndarray] = None
        self.reference_covs_: Optional[List[ArrayLike]] = None
        self.generalized_filters_: Optional[List[Dict]] = None
        self.generalized_templates_: Optional[List[Dict]] = None
        self.subject_models_: Optional[List[Dict]] = None
        self.target_covariance_diagnostics_: Optional[List[Dict[str, float]]] = None
        self.target_alignment_mode_used_: Optional[str] = None
        self.target_alignment_status_: Optional[str] = None
        self.target_alignment_reason_: Optional[str] = None
        self.source_score_diagnostics_: Optional[List[Dict[str, object]]] = None
        self.fitted_: bool = False

    def _safe_corrcoef(self, a, b, eps=1e-12):
        """
        计算一维相关系数，带数值保护，返回[-1,1]
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()

        if a.size == 0 or b.size == 0:
            return 0.0

        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]

        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

        a = a - np.mean(a)
        b = b - np.mean(b)

        da = np.linalg.norm(a)
        db = np.linalg.norm(b)
        if da < eps or db < eps:
            return 0.0

        c = float(np.dot(a, b) / (da * db))
        return float(np.clip(c, -1.0, 1.0))


    def _domain_similarity_score(
        self,
        X_src: ArrayLike,
        y_src: ArrayLike,
        X_tar: ArrayLike,
        y_tar: ArrayLike,
        classes: ArrayLike,
    ) -> float:
        """
        源域-目标域相似性分数（0~1）：
        - 对每个类别，计算源模板与目标模板相关性
        - 按可用类别平均
        """
        X_src = _ensure_3d(X_src)
        X_tar = _ensure_3d(X_tar)
        y_src = np.asarray(y_src)
        y_tar = np.asarray(y_tar)
        classes = np.asarray(classes)

        sims = []
        for cls in classes:
            idx_s = np.where(y_src == cls)[0]
            idx_t = np.where(y_tar == cls)[0]
            if idx_s.size == 0 or idx_t.size == 0:
                continue

            T_src = X_src[idx_s].mean(axis=0)  # (n_ch, n_samp)
            T_tar = X_tar[idx_t].mean(axis=0)

            c = self._safe_corrcoef(T_src, T_tar)  # [-1,1]
            sims.append((c + 1.0) / 2.0)          # -> [0,1]

        if len(sims) == 0:
            return 0.0
        return float(np.mean(sims))

    def _source_quality_score(
        self,
        X_src: ArrayLike,
        y_src: ArrayLike,
        X_tar: Optional[ArrayLike],
        y_tar: Optional[ArrayLike],
        classes: Sequence,
        src_stability: Optional[float] = None,
        tar_stability: Optional[float] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> float:
        """
        Compute the legacy multiplicative source-quality score q_m.

        This helper is retained for the explicitly named ``legacy_multiplicative``
        ablation and compatibility aliases. The revision's primary
        ``similarity_confidence`` mode is implemented in ``_compute_source_score``
        and uses target-source similarity with a bounded STC penalty.

        - n_calib = 1:
            q_m = Sim_{m,t}^{tmpl} * sqrt(STC_m)
        - n_calib >= 2:
            q_m = Sim_{m,t}^{tmpl} * sqrt(STC_m * STC_t)
        - n_calib = 0:
            q_m = STC_m
        """
        if src_stability is None:
            stc_src = _split_half_template_stability(
                X_src, y_src, classes,
                random_state=self.random_state if random_state is None else random_state,
                n_repeats=n_repeats,
                split_mode=self.stc_split_mode,
            )
        else:
            stc_src = float(src_stability)

        has_target_calib = (
            X_tar is not None and
            y_tar is not None and
            len(y_tar) > 0
        )

        if not has_target_calib:
            return float(np.clip(stc_src, 0.0, None))

        sim = _subject_template_similarity(X_src, y_src, X_tar, y_tar, classes)

        target_counts = [np.sum(np.asarray(y_tar) == cls) for cls in classes]
        has_target_split_half = bool(target_counts) and min(target_counts) >= 2
        if not has_target_split_half:
            return float(np.clip(sim * np.sqrt(max(stc_src, 0.0)), 0.0, None))

        if tar_stability is None:
            stc_tar = _split_half_template_stability(
                X_tar, y_tar, classes,
                random_state=self.random_state if random_state is None else random_state,
                n_repeats=n_repeats,
                split_mode=self.stc_split_mode,
            )
        else:
            stc_tar = float(tar_stability)

        # Low-sample target STC is noisy.  Shrink it toward a neutral
        # reliability of one and introduce it progressively as calibration
        # support grows, while retaining both source and target stability.
        stc_src = float(np.clip(stc_src, 1e-4, 1.0))
        stc_tar = float(np.clip(stc_tar, 1e-4, 1.0))
        n_target = int(min(target_counts))
        gate = float(np.clip((n_target - 1) / 3.0, 0.0, 1.0))
        stc_tar_gated = (1.0 - gate) + gate * stc_tar
        q_m = sim * np.sqrt(stc_src * stc_tar_gated)
        return float(np.clip(q_m, 0.0, None))

    def _source_score_components(self, X_src, y_src, X_tar, y_tar, src_stability, classes):
        """Return a score plus audit components without changing score semantics."""
        mode = self.source_score_mode
        has_target_calib = X_tar is not None and y_tar is not None and len(y_tar) > 0
        target_counts = [] if not has_target_calib else [int(np.sum(np.asarray(y_tar) == cls)) for cls in classes]
        n_target = int(min(target_counts)) if target_counts else 0
        unavailable = "no_target_calibration" if not has_target_calib else None
        components = {
            "source_score_mode_requested": mode,
            "source_score_mode_effective": mode if has_target_calib else "stability_fallback",
            "stc_split_mode": self.stc_split_mode,
            "source_stability": float(src_stability),
            "template_similarity": None,
            "target_min_class_count": n_target if has_target_calib else None,
            "target_counts_by_class": target_counts if has_target_calib else None,
            "confidence_lambda": None,
            "confidence_uncertainty": None,
            "confidence_gate": None,
            "confidence_factor": None,
            "target_stability": None,
            "target_stability_gate": None,
            "target_dependent_state": "available" if has_target_calib else "unavailable",
            "target_dependent_reason": unavailable,
        }
        if mode in {"adaptive", "robust", "legacy_multiplicative"}:
            if not has_target_calib:
                return float(max(src_stability, 0.0)), components
            similarity = float(_subject_template_similarity(X_src, y_src, X_tar, y_tar, classes))
            components["template_similarity"] = similarity
            if n_target < 2:
                return float(np.clip(similarity * np.sqrt(max(src_stability, 0.0)), 0.0, None)), components
            target_stability = float(_split_half_template_stability(
                X_tar, y_tar, classes, random_state=self.random_state, n_repeats=5,
                split_mode=self.stc_split_mode,
            ))
            target_gate = float(np.clip((n_target - 1) / 3.0, 0.0, 1.0))
            components["target_stability"] = target_stability
            components["target_stability_gate"] = target_gate
            stc_src = float(np.clip(src_stability, 1e-4, 1.0))
            stc_tar = float(np.clip(target_stability, 1e-4, 1.0))
            return float(np.clip(similarity * np.sqrt(stc_src * ((1.0 - target_gate) + target_gate * stc_tar)), 0.0, None)), components
        if mode == "similarity_confidence":
            if not has_target_calib:
                return float(max(src_stability, 0.0)), components
            similarity = float(self._domain_similarity_score(X_src, y_src, X_tar, y_tar, classes))
            components["template_similarity"] = similarity
            if n_target <= 0:
                return float(max(similarity, 0.0)), components
            stability = float(np.clip(src_stability, 0.0, 1.0))
            uncertainty = 1.0 - stability
            gate = float(np.clip((n_target - 1) / 2.0, 0.0, 1.0))
            confidence_factor = 1.0 - self.confidence_lambda * gate * uncertainty
            components.update({
                "confidence_lambda": float(self.confidence_lambda),
                "confidence_uncertainty": uncertainty,
                "confidence_gate": gate,
                "confidence_factor": confidence_factor,
            })
            return float(max(similarity * confidence_factor, 0.0)), components
        if mode == "stability":
            return float(max(src_stability, 0.0)), components
        if mode == "similarity":
            if not has_target_calib:
                return float(max(src_stability, 0.0)), components
            similarity = float(self._domain_similarity_score(X_src, y_src, X_tar, y_tar, classes))
            components["template_similarity"] = similarity
            return similarity, components
        if mode == "random":
            if not hasattr(self, "_random_score_rng"):
                self._random_score_rng = np.random.RandomState(self.random_state)
            return float(self._random_score_rng.rand()), components
        raise ValueError(f"Unknown source_score_mode={mode}")

    def _compute_source_score(self, X_src, y_src, X_tar, y_tar, src_stability, classes):
        """Compatibility wrapper for the scoring value used by source selection."""
        score, _ = self._source_score_components(X_src, y_src, X_tar, y_tar, src_stability, classes)
        return score

    def fit(
        self,
        X_source: ArrayLike,
        y_source: ArrayLike,
        subjects_source: Sequence,
        target_calib_X: Optional[ArrayLike] = None,
        target_calib_y: Optional[ArrayLike] = None,
    ) -> "SourceAlignedDualTemplateSSVEP":
        """
        训练模型。
        可选传入少量目标域标注校准数据，用于更准确源域选择。
        """
        X_source = _ensure_3d(X_source)
        y_source = np.asarray(y_source)
        subjects_source = np.asarray(subjects_source)

        if len(X_source) != len(y_source) or len(X_source) != len(subjects_source):
            raise ValueError("X_source, y_source, and subjects_source must have the same length.")

        if target_calib_X is not None:
            target_calib_X = _ensure_3d(target_calib_X)
            if target_calib_y is None:
                raise ValueError("target_calib_y must be provided when target_calib_X is given.")
            target_calib_y = np.asarray(target_calib_y)
            if len(target_calib_X) != len(target_calib_y):
                raise ValueError("target_calib_X and target_calib_y must have the same length.")

        self.classes_ = np.unique(y_source)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}

        # 滤波器组展开：得到 (n_bands, n_trials, n_channels, n_samples)
        X_source_fb = self._apply_filterbank(X_source)
        self.n_bands_ = X_source_fb.shape[0]
        self.band_weights_ = self._resolve_band_weights(self.n_bands_)

        subject_ids = np.unique(subjects_source)
        if subject_ids.size < 1:
            raise ValueError("No source subjects were provided.")

        source_models_by_band: List[Dict] = [dict() for _ in range(self.n_bands_)]
        raw_subject_templates: Dict = {}
        raw_subject_stability: Dict = {}

        # ========== Stage 1：逐源被试、逐频带学习局部滤波器与模板 ==========
        for subject in subject_ids:
            idx_subject = np.where(subjects_source == subject)[0]
            X_sub_raw = X_source[idx_subject]
            y_sub = y_source[idx_subject]

            raw_subject_stability[subject] = _split_half_template_stability(
                X_sub_raw, y_sub, self.classes_, random_state=self.random_state, n_repeats=5,
                split_mode=self.stc_split_mode,
            )
            raw_subject_templates[subject] = {
                cls: X_sub_raw[y_sub == cls].mean(axis=0) for cls in self.classes_ if np.any(y_sub == cls)
            }

            for band_idx in range(self.n_bands_):
                X_sub = X_source_fb[band_idx, idx_subject]
                templates_raw = {}
                w_filters = {}
                for cls in self.classes_:
                    class_idx = np.where(y_sub == cls)[0]
                    if class_idx.size == 0:
                        continue
                    templates_raw[cls] = X_sub[class_idx].mean(axis=0)

                    # Stage1开关：关闭时用均匀向量替代
                    if self.enable_stage1:
                        w_filters[cls] = self._learn_stage1_filter(X_sub, y_sub, cls)
                    else:
                        n_channels = X_sub.shape[1]
                        w_filters[cls] = np.ones(n_channels, dtype=float) / np.sqrt(n_channels)

                covariance = _mean_covariance(X_sub, reg=self.reg)
                source_models_by_band[band_idx][subject] = _BandSubjectModel(
                    subject=subject,
                    covariance=covariance,
                    align_matrix=None,
                    stability=raw_subject_stability[subject],
                    templates_raw=templates_raw,
                    templates_aligned={},
                    w_filters=w_filters,
                )

        # ========== 源被试选择 ==========
        score_components = []
        for subject in subject_ids:
            mask = subjects_source == subject
            score, components = self._source_score_components(
                X_src=X_source[mask],
                y_src=y_source[mask],
                X_tar=target_calib_X,
                y_tar=target_calib_y,
                src_stability=raw_subject_stability[subject],
                classes=self.classes_,
            )
            components["candidate_subject"] = int(subject)
            components["raw_score_before_nonpositive_fallback"] = float(score)
            score_components.append(components)
        # Target calibration is optional: without it, source selection uses
        # the source-stability fallback recorded in the diagnostics above.

        raw_scores = np.asarray(
            [record["raw_score_before_nonpositive_fallback"] for record in score_components], dtype=float
        )
        all_nonpositive_fallback = bool(np.all(raw_scores <= 0))
        if all_nonpositive_fallback:
            raw_scores = np.ones_like(raw_scores)
        # Lexicographic sorting makes ties reproducible across NumPy versions.
        order = np.lexsort((np.asarray(subject_ids, dtype=int), -raw_scores))
        n_selected = len(order) if self.n_sources is None else max(1, min(self.n_sources, len(order)))
        selected_order = order[:n_selected]
        selected_subjects = subject_ids[selected_order]
        selected_scores = raw_scores[selected_order]

        # 源权重模式
        if self.source_weight_mode == "score":
            selected_weights = selected_scores / (selected_scores.sum() + 1e-12)
        elif self.source_weight_mode == "uniform":
            selected_weights = np.ones_like(selected_scores, dtype=float) / len(selected_scores)
        else:
            raise ValueError(f"Unknown source_weight_mode={self.source_weight_mode}")

        selected_weight_by_subject = {int(subject): float(weight) for subject, weight in zip(selected_subjects, selected_weights)}
        rank_by_index = {int(index): rank + 1 for rank, index in enumerate(order)}
        for index, record in enumerate(score_components):
            subject = int(record["candidate_subject"])
            record.update({
                "raw_score": float(raw_scores[index]),
                "all_nonpositive_score_fallback": all_nonpositive_fallback,
                "candidate_rank": rank_by_index[index],
                "selected": subject in selected_weight_by_subject,
                "selected_weight": selected_weight_by_subject.get(subject),
            })
        self.source_score_diagnostics_ = sorted(score_components, key=lambda record: record["candidate_rank"])

        self.selected_subjects_ = selected_subjects
        self.source_scores_ = selected_scores
        self.source_weights_ = selected_weights

        # ========== Stage 2：协方差对齐 + 跨被试广义模板 ==========
        self.reference_covs_ = []
        self.subject_models_ = []
        self.generalized_templates_ = []
        self.generalized_filters_ = []

        for band_idx in range(self.n_bands_):
            band_models = source_models_by_band[band_idx]

            # 参考协方差：选中源被试协方差均值
            ref_cov = np.mean([band_models[s].covariance for s in selected_subjects], axis=0)
            ref_cov = 0.5 * (ref_cov + ref_cov.T)
            self.reference_covs_.append(ref_cov)

            aligned_models = {}

            if self.enable_stage2:
                sqrt_ref = _sqrtm_spd(ref_cov)
                inv_sqrt_source = {s: _inv_sqrtm_spd(band_models[s].covariance) for s in selected_subjects}

                for subject in selected_subjects:
                    model = band_models[subject]
                    A = sqrt_ref @ inv_sqrt_source[subject]
                    templates_aligned = {cls: A @ template for cls, template in model.templates_raw.items()}
                    aligned_models[subject] = _BandSubjectModel(
                        subject=subject,
                        covariance=model.covariance,
                        align_matrix=A,
                        stability=model.stability,
                        templates_raw=model.templates_raw,
                        templates_aligned=templates_aligned,
                        w_filters=model.w_filters,
                    )
            else:
                # 关闭Stage2：不做EA，模板直接使用raw
                for subject in selected_subjects:
                    model = band_models[subject]
                    aligned_models[subject] = _BandSubjectModel(
                        subject=subject,
                        covariance=model.covariance,
                        align_matrix=np.eye(model.covariance.shape[0]),
                        stability=model.stability,
                        templates_raw=model.templates_raw,
                        templates_aligned=dict(model.templates_raw),
                        w_filters=model.w_filters,
                    )

            self.subject_models_.append(aligned_models)

            generalized_templates = {}
            generalized_filters = {}
            for cls in self.classes_:
                available_subjects = [s for s in selected_subjects if cls in aligned_models[s].templates_aligned]
                if not available_subjects:
                    continue
                weights = np.asarray(
                    [selected_weights[np.where(selected_subjects == s)[0][0]] for s in available_subjects]
                )
                weights = weights / (weights.sum() + 1e-12)

                template = np.sum(
                    [w * aligned_models[s].templates_aligned[cls] for s, w in zip(available_subjects, weights)],
                    axis=0,
                )
                generalized_templates[cls] = template
                generalized_filters[cls] = self._learn_cross_subject_filter(
                    aligned_models, available_subjects, weights, cls
                )
            self.generalized_templates_.append(generalized_templates)
            self.generalized_filters_.append(generalized_filters)

        self.fitted_ = True
        return self

    def predict(
        self,
        X: ArrayLike,
        calib_X: Optional[ArrayLike] = None,
        return_scores: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测标签。
        三分支融合得分：
        1) 与谐波参考 Yf 的 CCA
        2) 与跨被试广义模板的一维相关
        3) 与各源被试专属模板的一维相关（按源权重加权）
        """
        self._check_is_fitted()

        harmonic_enabled = self.enable_branch_r1 and self.enable_harmonic_branch
        if not (harmonic_enabled or self.enable_branch_r2 or self.enable_branch_r3):
            raise ValueError("At least one branch must be enabled.")

        X = _ensure_3d(X)
        n_test = X.shape[0]

        X_fb = self._apply_filterbank(X)

        # Stage 2 target alignment is explicit. The default uses only labeled
        # calibration data; transductive test-batch alignment is opt-in.
        self.target_covariance_diagnostics_ = []
        self.target_alignment_mode_used_ = "none"
        self.target_alignment_status_ = "skipped"
        self.target_alignment_reason_ = "not_evaluated"
        if self.enable_stage2 and self.target_alignment_mode != "none":
            if calib_X is not None:
                calib_X = _ensure_3d(calib_X)
                calib_fb = self._apply_filterbank(calib_X)
            else:
                calib_fb = None

            if self.target_alignment_mode == "calibration" and calib_fb is not None:
                aligned_fb = self._align_target_batch(X_fb, covariance_fb=calib_fb)
                self.target_alignment_mode_used_ = "calibration"
                self.target_alignment_status_ = "performed"
                self.target_alignment_reason_ = "calibration_only_covariance"
            elif self.target_alignment_mode == "transductive":
                covariance_fb = X_fb if calib_fb is None else np.concatenate([X_fb, calib_fb], axis=1)
                aligned_fb = self._align_target_batch(X_fb, covariance_fb=covariance_fb)
                self.target_alignment_mode_used_ = "transductive"
                self.target_alignment_status_ = "performed"
                self.target_alignment_reason_ = "transductive_covariance"
            elif self.target_alignment_mode == "calibration":
                aligned_fb = X_fb
                self.target_alignment_status_ = "not_applicable_no_calibration"
                self.target_alignment_reason_ = "no_target_calibration"
            else:
                raise ValueError(f"Unknown target_alignment_mode={self.target_alignment_mode}")
        else:
            aligned_fb = X_fb
            self.target_alignment_status_ = "disabled"
            self.target_alignment_reason_ = "stage2_disabled_or_none"

        scores = np.zeros((n_test, len(self.classes_)), dtype=float)
        for trial_idx in range(n_test):
            for cls_idx, cls in enumerate(self.classes_):
                r1 = 0.0
                r2 = 0.0
                r3 = 0.0
                for band_idx in range(self.n_bands_):
                    wb = self.band_weights_[band_idx]
                    X_trial = aligned_fb[band_idx, trial_idx]

                    if harmonic_enabled and (self.Yf is not None):
                        ref = self.Yf[self.class_to_index_[cls]]
                        c1 = _fbcca_corr(X_trial,ref,srate=self.srate,n_bands=self.n_bands_)
                    else:
                        c1 = 0.0

                    # 分支2：跨被试广义模板
                    if self.enable_branch_r2:
                        v = self.generalized_filters_[band_idx].get(cls)
                        Z = self.generalized_templates_[band_idx].get(cls)
                        if v is None or Z is None:
                            c2 = 0.0
                        else:
                            c2 = _corr_1d(v @ X_trial, v @ Z)
                    else:
                        c2 = 0.0

                    # 分支3：源被试专属模板集合
                    if self.enable_branch_r3:
                        c3 = 0.0
                        for subject, weight in zip(self.selected_subjects_, self.source_weights_):
                            model = self.subject_models_[band_idx][subject]
                            w = model.w_filters.get(cls)
                            T = model.templates_aligned.get(cls)
                            if w is None or T is None:
                                continue
                            c3 += weight * _corr_1d(w @ X_trial, w @ T)
                    else:
                        c3 = 0.0

                    r1 += wb * c1
                    r2 += wb * c2
                    r3 += wb * c3

                # 融合模式
                if self.fusion_mode == "signed_square":
                    score = (
                        np.sign(r1) * (r1 ** 2)
                        + np.sign(r2) * (r2 ** 2)
                        + np.sign(r3) * (r3 ** 2)
                    )
                elif self.fusion_mode == "plain_sum":
                    score = r1 + r2 + r3
                elif self.fusion_mode == "abs_sum":
                    score = abs(r1) + abs(r2) + abs(r3)
                else:
                    raise ValueError(f"Unknown fusion_mode={self.fusion_mode}")

                scores[trial_idx, cls_idx] = score

        pred_idx = np.argmax(scores, axis=1)
        y_pred = self.classes_[pred_idx]
        if return_scores:
            return y_pred, scores
        return y_pred


    def fit_predict_subject(
        self,
        X_group, y_group, subjects,
        target_subject,
        n_calib=0,
        random_state=42,
        return_scores=False,
    ):
        X_group = _ensure_3d(X_group)
        y_group = np.asarray(y_group)
        subjects = np.asarray(subjects)

        src_mask = subjects != target_subject
        tar_mask = subjects == target_subject

        X_src, y_src, s_src = X_group[src_mask], y_group[src_mask], subjects[src_mask]
        X_tar, y_tar = X_group[tar_mask], y_group[tar_mask]

        if n_calib > 0:
            calib_idx, test_idx = self._split_target_calibration(
                y_tar, n_calib, random_state
            )
            X_calib, y_calib = X_tar[calib_idx], y_tar[calib_idx]
            # Keep calibration trials out of evaluation.
            X_test, y_test = X_tar[test_idx], y_tar[test_idx]
        else:
            X_calib = y_calib = None
            X_test, y_test = X_tar, y_tar

        self.fit(X_src, y_src, s_src,
                target_calib_X=X_calib, target_calib_y=y_calib)
        if n_calib > 0:
            pred_out = self.predict(X_test, calib_X=X_calib, return_scores=return_scores)
        else:
            pred_out = self.predict(X_test, return_scores=return_scores)

        if return_scores:
            y_pred, scores = pred_out
            return y_pred, y_test, scores
        return pred_out, y_test
    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _check_is_fitted(self) -> None:
        """训练状态检查。"""
        if not self.fitted_:
            raise RuntimeError("The model has not been fitted yet.")

    def _resolve_band_weights(self, n_bands: int) -> np.ndarray:
        """解析并归一化滤波器组权重。"""
        if self.filterweights is None:
            return np.ones(n_bands, dtype=float) / n_bands
        weights = np.asarray(self.filterweights, dtype=float)
        if weights.shape[0] != n_bands:
            raise ValueError(
                f"filterweights length ({weights.shape[0]}) does not match number of bands ({n_bands})."
            )
        denom = weights.sum()
        if denom <= 0:
            return np.ones(n_bands, dtype=float) / n_bands
        return weights / denom

    def _apply_filterbank(self, X: ArrayLike) -> ArrayLike:
        """对输入应用滤波器组，输出形状：(n_bands, n_trials, n_channels, n_samples)。"""
        X = _ensure_3d(X)
        if self.filterbank is None:
            return X[None, ...]
        out = []
        for fb in self.filterbank:
            out.append(self._apply_single_filter(X, fb))
        return np.stack(out, axis=0)

    def _apply_single_filter(self, X: ArrayLike, filt) -> ArrayLike:
        """支持 (b,a) 或 SOS 两种滤波器格式。"""
        X = np.asarray(X, dtype=float)
        if isinstance(filt, (tuple, list)) and len(filt) == 2:
            b, a = filt
            return signal.filtfilt(b, a, X, axis=-1)
        filt_arr = np.asarray(filt)
        if filt_arr.ndim == 2 and filt_arr.shape[1] == 6:
            return signal.sosfiltfilt(filt_arr, X, axis=-1)
        raise ValueError(
            "Unsupported filter-bank entry. Each filter must be an SOS array or a (b, a) tuple."
        )

    def _neighbor_classes(self, cls) -> List:
        """Return the current class and nearest stimulus-frequency neighbours."""
        ordered_classes = list(self.classes_)
        if self.freqs is not None and len(self.freqs) == len(self.classes_):
            ordered_classes = [
                item[1] for item in sorted(
                    zip(self.freqs, self.classes_), key=lambda item: item[0]
                )
            ]
        idx = ordered_classes.index(cls)
        neighbors = [cls]
        for radius in range(1, self.neighbor_radius + 1):
            left = idx - radius
            right = idx + radius
            if left >= 0:
                neighbors.append(ordered_classes[left])
            if right < len(ordered_classes):
                neighbors.append(ordered_classes[right])
        return neighbors

    def _neighbor_weight(self, cls, neighbor_cls) -> float:
        """Distance-decayed neighbour weight; self-class always has weight one."""
        if cls == neighbor_cls:
            return 1.0
        if self.freqs is not None and len(self.freqs) == len(self.classes_):
            class_to_frequency = dict(zip(self.classes_, self.freqs))
            ordered_frequencies = np.sort(np.asarray(self.freqs, dtype=float))
            frequency_steps = np.diff(ordered_frequencies)
            scale = float(np.median(frequency_steps[frequency_steps > 0])) if np.any(frequency_steps > 0) else 1.0
            distance = abs(class_to_frequency[cls] - class_to_frequency[neighbor_cls]) / scale
        else:
            ordered_classes = list(self.classes_)
            distance = abs(ordered_classes.index(cls) - ordered_classes.index(neighbor_cls))
        return float(self.neighbor_strength * np.exp(-self.neighbor_decay * distance))

    def _learn_stage1_filter(self, X_sub: ArrayLike, y_sub: ArrayLike, cls) -> ArrayLike:
        """
        Stage1：针对单个源被试、单个类别学习滤波器。
        使用目标类别+邻居类别构造 S/Q，再做广义特征分解。
        """
        n_channels = X_sub.shape[1]
        S = np.zeros((n_channels, n_channels), dtype=float)
        Q = np.zeros((n_channels, n_channels), dtype=float)

        for neighbor_cls in self._neighbor_classes(cls):
            idx = np.where(y_sub == neighbor_cls)[0]
            if idx.size == 0:
                continue
            weight = self._neighbor_weight(cls, neighbor_cls)
            S_cls, Q_cls = _trial_x_trial_sum(X_sub[idx])
            S += weight * S_cls
            Q += weight * Q_cls

        if np.allclose(S, 0) or np.allclose(Q, 0):
            return np.ones(n_channels, dtype=float) / np.sqrt(n_channels)
        return _top_generalized_eigenvector(S, Q, reg=self.reg)

    def _learn_cross_subject_filter(
        self,
        aligned_models: Dict,
        available_subjects: Sequence,
        weights: np.ndarray,
        cls,
    ) -> ArrayLike:
        """
        学习跨被试滤波器：
        - R: 各被试模板协方差加权和
        - P: 各被试模板之间互相关加权和
        """
        n_channels = next(iter(aligned_models.values())).templates_aligned[cls].shape[0]
        P = np.zeros((n_channels, n_channels), dtype=float)
        R = np.zeros((n_channels, n_channels), dtype=float)

        for s, ws in zip(available_subjects, weights):
            template_s = aligned_models[s].templates_aligned[cls]
            R += ws * _cov_spd(template_s, reg=self.reg)
        for i, s1 in enumerate(available_subjects):
            for j, s2 in enumerate(available_subjects):
                if i == j:
                    continue
                Xi = _center_rows(aligned_models[s1].templates_aligned[cls])
                Xj = _center_rows(aligned_models[s2].templates_aligned[cls])
                P += weights[i] * weights[j] * (Xi @ Xj.T) / max(Xi.shape[1] - 1, 1)

        if np.allclose(P, 0) or np.allclose(R, 0):
            return np.ones(n_channels, dtype=float) / np.sqrt(n_channels)
        return _top_generalized_eigenvector(P, R, reg=self.reg)

    def _align_target_batch(self, X_fb: ArrayLike, covariance_fb: Optional[ArrayLike] = None) -> ArrayLike:
        """
        将目标 trial 对齐到每个频带的参考协方差空间：
        A_t = C_ref^{1/2} C_t^{-1/2}

        covariance_fb controls which unlabeled/labeled trials estimate C_t.
        It is separate from X_fb so inductive calibration-only alignment does
        not consume test trials.
        """
        if covariance_fb is None:
            covariance_fb = X_fb
        aligned = np.zeros_like(X_fb)
        for band_idx in range(self.n_bands_):
            trials = X_fb[band_idx]
            C_t = _mean_covariance(covariance_fb[band_idx], reg=self.reg)
            trial_covariances = _trial_covariances(covariance_fb[band_idx], reg=self.reg)
            A_t = _sqrtm_spd(self.reference_covs_[band_idx]) @ _inv_sqrtm_spd(C_t)
            aligned[band_idx] = np.einsum("ab,tbs->tas", A_t, trials)
            C_after = A_t @ C_t @ A_t.T
            C_ref = self.reference_covs_[band_idx]
            reference_norm = float(np.linalg.norm(C_ref, ord="fro"))
            denominator = max(reference_norm, 1e-12)
            before = float(np.linalg.norm(C_t - C_ref, ord="fro"))
            after = float(np.linalg.norm(C_after - C_ref, ord="fro"))
            trial_deviations = np.asarray(
                [np.linalg.norm(covariance - C_t, ord="fro") for covariance in trial_covariances], dtype=float
            )
            finite = bool(np.isfinite(C_t).all() and np.isfinite(C_ref).all() and np.isfinite(C_after).all())
            self.target_covariance_diagnostics_.append({
                "band": int(band_idx),
                "status": "available",
                "reason": "calibration_covariance" if covariance_fb is not X_fb else "test_or_transductive_covariance",
                "n_covariance_trials": int(len(covariance_fb[band_idx])),
                "n_channels": int(C_t.shape[0]),
                "condition_number_target": float(np.linalg.cond(C_t)),
                "condition_number_reference": float(np.linalg.cond(C_ref)),
                "effective_rank_target": _entropy_effective_rank(C_t),
                "effective_rank_reference": _entropy_effective_rank(C_ref),
                "frobenius_distance_before": before,
                "frobenius_distance_after": after,
                "normalized_frobenius_distance_before": before / denominator,
                "normalized_frobenius_distance_after": after / denominator,
                "frobenius_distance_improvement": before - after,
                "alignment_residual_frobenius": after,
                "trial_covariance_deviation_mean": float(np.mean(trial_deviations)),
                "trial_covariance_deviation_rms": float(np.sqrt(np.mean(trial_deviations ** 2))),
                "trial_covariance_deviation_q95": float(np.quantile(trial_deviations, 0.95)),
                "trial_covariance_deviation_mean_normalized": float(np.mean(trial_deviations)) / denominator,
                "finite": finite,
                "spd_eigendecomposition_ok": finite,
            })
        return aligned

    def _split_target_calibration(
        self,
        y_target: np.ndarray,
        n_calib: int,
        random_state: int = 42,
    ):
        """
        按类别从目标被试中抽取校准样本（每类 n_calib 个），返回校准/测试索引。
        """
        y_target = np.asarray(y_target)
        n = len(y_target)
        all_idx = np.arange(n)

        if n_calib <= 0:
            return np.array([], dtype=int), all_idx

        rng = np.random.default_rng(random_state)
        calib_idx = []

        # 关键修复：不用 self.classes_，改用当前目标标签里的类别
        classes = np.unique(y_target)

        for cls in classes:
            idx = np.where(y_target == cls)[0]
            if idx.size == 0:
                continue
            pick = min(n_calib, idx.size)
            calib_idx.extend(rng.choice(idx, size=pick, replace=False).tolist())

        calib_idx = np.array(sorted(set(calib_idx)), dtype=int)
        test_idx = np.setdiff1d(all_idx, calib_idx)

        return calib_idx, test_idx
