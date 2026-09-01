"""IISTLF
"""

import numpy as np
from scipy.linalg import eigh
from scipy import signal
from scipy.signal import butter, filtfilt


_CHUNK = 128


_DTYPE = np.float32


def _sign_square(r: float) -> float:
    return float(np.sign(r) * (r ** 2))


def trca_spatial_filter(X):
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_ch, n_samples = X.shape
    X = np.nan_to_num(X)
    Xc = X - X.mean(axis=2, keepdims=True)

    if n_trials < 2:
        C = np.einsum("tcs,tds->cd", Xc, Xc, optimize=True) / max(n_trials * (n_samples - 1), 1)
        vals, vecs = np.linalg.eigh(C + 1e-8 * np.eye(n_ch))
        return vecs[:, np.argmax(vals)]

    Q = np.einsum("tcs,tds->cd", Xc, Xc, optimize=True) / max(n_trials * (n_samples - 1), 1)
    sum_all = np.sum(Xc, axis=0)
    auto = np.einsum("tcs,tds->cd", Xc, Xc, optimize=True)
    cross = (sum_all @ sum_all.T) - auto
    S = cross / max(n_trials * (n_trials - 1) * (n_samples - 1), 1)
    vals, vecs = eigh(S + 1e-8 * np.eye(n_ch), Q + 1e-6 * np.eye(n_ch))
    return vecs[:, np.argmax(vals)]


def whiten_trial(X):
    C = np.cov(X)
    d, V = np.linalg.eigh(C + 1e-8 * np.eye(C.shape[0]))
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(d, 1e-10))) @ V.T
    return W @ X, W


def _inv_sqrt_psd(M: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """批量对称 PSD 矩阵的逆平方根，M 形状 (..., n, n)。"""
    d, V = np.linalg.eigh(M + reg * np.eye(M.shape[-1]))
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-10))
    return (V * dinv[..., None, :]) @ np.swapaxes(V, -1, -2)


class IISTLF:

    def __init__(self, srate, freqs, n_harmonics=3, n_subbands=2, a=1.25, b=0.25, tpl_ds=4):
        self.srate = float(srate)
        self.freqs = np.array(freqs, dtype=float)
        self.n_harmonics = int(n_harmonics)
        self.n_subbands = int(n_subbands)
        self.a, self.b = float(a), float(b)
        self.tpl_ds = int(max(1, tpl_ds))

    def _build_refs(self, n_samples):
        t = np.arange(1, n_samples + 1) / self.srate
        refs = []
        for f in self.freqs:
            Y = []
            for h in range(1, self.n_harmonics + 1):
                Y += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
            refs.append(np.asarray(Y, float))  # (2H, Ns)
        return refs

    def _build_subbands(self):
        lows = [8.0 + 14.0 * k for k in range(self.n_subbands)]
        highs = [min(l + 14.0, self.srate / 2 - 1.0) for l in lows]
        return list(zip(lows, highs))

    def _design_fb_iir(self):
        passband = [8, 18, 26, 34, 42, 50, 58, 66, 74, 82]
        stopband = [6, 14, 20, 28, 36, 44, 52, 60, 68, 76]

        nyq = self.srate / 2.0
        n_sb = min(self.n_subbands, len(passband))

        sos_list = []
        for i in range(n_sb):
            wp = [passband[i] / nyq, min(90 / nyq, 0.999)]
            ws = [stopband[i] / nyq, min(100 / nyq, 0.999)]

            wp[0] = max(wp[0], 1e-4)
            ws[0] = max(ws[0], 1e-4)
            if not (0 < ws[0] < wp[0] < wp[1] < ws[1] <= 1):
                hi_wp = min(0.90, 0.95)
                hi_ws = min(0.95, 0.99)
                wp = [max(passband[i] / nyq, 1e-3), hi_wp]
                ws = [max(stopband[i] / nyq, 5e-4), hi_ws]

            n, wn = signal.cheb1ord(wp, ws, gpass=2, gstop=40)
            sos = signal.cheby1(n, rp=0.1, Wn=wn, btype="bandpass", output="sos")
            sos_list.append(sos)

        return sos_list

    def _fast_fb(self, X):
        # 批量切比雪夫滤波器组：X 为 (n_ch, Ns) 或 (T, n_ch, Ns)，沿时间轴滤波
        sos_list = getattr(self, "_fb_sos_fast", self.fb_sos)
        return [signal.sosfiltfilt(sos, X, axis=-1) for sos in sos_list]

    def fit(self, source_Xy, target_calib_X, target_calib_y):
        Xs, ys = source_Xy
        Xs = np.asarray(Xs, float)
        ys = np.asarray(ys, int)
        Xt = np.asarray(target_calib_X, float)
        yt = np.asarray(target_calib_y, int)

        if Xt.ndim != 3:
            raise ValueError(f"target_calib_X must be (n_trials, n_ch, n_samples), got {Xt.shape}.")
        if len(Xt) != len(yt):
            raise ValueError(f"target_calib_X and target_calib_y length mismatch: {len(Xt)} vs {len(yt)}.")

        _, n_ch, n_samples = Xs.shape
        if Xt.shape[1:] != (n_ch, n_samples):
            raise ValueError(
                f"target_calib_X shape {Xt.shape[1:]} does not match source data {(n_ch, n_samples)}."
            )
        self.n_ch_ = int(n_ch)
        self.n_samples_ = int(n_samples)

        self.Yf = self._build_refs(n_samples)

        self.subbands = self._build_subbands()
        self.wfb = np.array([(k + 1) ** (-self.a) + self.b for k in range(self.n_subbands)], float)
        self.fb_sos = self._design_fb_iir()
        self.n_subbands = len(self.fb_sos)
        self.wfb = np.array([(k + 1) ** (-self.a) + self.b for k in range(self.n_subbands)], float)

        self.source_models = {}
        for cls in range(len(self.freqs)):
            Xc = Xs[ys == cls]
            if len(Xc) < 2:
                continue
            w = trca_spatial_filter(Xc)
            T = (Xc.transpose(0, 2, 1) @ w).mean(axis=0)[:: self.tpl_ds]
            self.source_models[cls] = {"w": w, "T": T}

        # 类平衡校准：每类独立收集，禁止单类指定
        classes = [c for c in range(len(self.freqs)) if np.any(yt == c)]
        self.classes_with_calib_ = classes
        Xt_by_class = {c: Xt[yt == c] for c in classes}
        self.n_calib_per_class_ = int(min((len(Xt_by_class[c]) for c in classes), default=0))

        if self.n_calib_per_class_ == 0:
            # 无目标校准：目标依赖组件退化为恒等映射（source-only fallback）
            self.calibration_status_ = "none"
            self.target_w_by_class = {}
            self.A_lst = np.eye(n_ch)
            self.Wt = np.eye(n_ch)
            self.L_cwa = np.eye(n_ch)
        else:
            self.calibration_status_ = "per_class"
            self.target_w_by_class = {
                c: trca_spatial_filter(Xt_by_class[c]) for c in classes
            }
            src_means, tar_means = [], []
            for c in classes:
                Xs_c = Xs[ys == c]
                if len(Xs_c) == 0:
                    continue
                src_means.append(Xs_c.mean(axis=0))
                tar_means.append(Xt_by_class[c].mean(axis=0))
            S_src = np.concatenate(src_means, axis=1)  # (n_ch, C*n_samples)
            S_tar = np.concatenate(tar_means, axis=1)
            self.A_lst = S_src @ S_tar.T @ np.linalg.pinv(S_tar @ S_tar.T + 1e-8 * np.eye(n_ch))
            # (3) 目标白化：由所有类别的目标类均值拼接估计，rho3 对测试试次使用同一 Wt
            _, self.Wt = whiten_trial(S_tar)
            
            # (4) 通道匹配：白化后的源/目标类均值之间做贪婪 1-1 匹配（逻辑与原实现一致）
            Xs_w, _ = whiten_trial(S_src)
            Xt_w = self.Wt @ S_tar
            M = np.abs(np.corrcoef(Xs_w, Xt_w)[:n_ch, n_ch:])
            L = np.zeros((n_ch, n_ch))
            ur, uc = set(), set()
            for i, j in np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0]:
                if i not in ur and j not in uc:
                    L[i, j] = 1.0
                    ur.add(i)
                    uc.add(j)
                if len(ur) == n_ch:
                    break
            self.L_cwa = L
        self._precompute_runtime_state()
        return self

    def _precompute_runtime_state(self):
        """fit 结束后预计算 predict 所需的全部固定量（与试次无关）。
        """
        C = len(self.freqs)
        n_ch = self.n_ch_
        eps = 1e-12

        # rho1：每类目标滤波器堆叠（缺校准的类别行保持零 → 贡献为 0）
        self._W1 = np.zeros((C, n_ch), dtype=_DTYPE)
        for c, w in self.target_w_by_class.items():
            self._W1[c] = w

        # rho2/rho3：源域滤波器与模板堆叠，并预乘对齐矩阵
        tpl_len = len(next(iter(self.source_models.values()))["T"]) if self.source_models else \
            int(np.ceil(self.n_samples_ / self.tpl_ds))
        self._W2 = np.zeros((C, n_ch), dtype=_DTYPE)
        self._Tpl = np.zeros((C, tpl_len), dtype=_DTYPE)
        for c, m in self.source_models.items():
            self._W2[c] = m["w"]
            self._Tpl[c] = m["T"]
        self._WA = (self._W2 @ self.A_lst.astype(_DTYPE)).astype(_DTYPE)
        self._WB = (self._W2 @ (self.L_cwa @ self.Wt).astype(_DTYPE)).astype(_DTYPE)

        Tc = self._Tpl - self._Tpl.mean(axis=1, keepdims=True)
        self._TplN = Tc / (np.linalg.norm(Tc, axis=1, keepdims=True) + eps)

        Yf = np.stack(self.Yf)  # (C, 2H, Ns) float64
        Yc64 = Yf - Yf.mean(axis=2, keepdims=True)
        self._Yc = Yc64.astype(_DTYPE)
        self._Ycn = (Yc64 / (np.linalg.norm(Yc64, axis=2, keepdims=True) + eps)).astype(_DTYPE)
        self._YcnT = np.ascontiguousarray(self._Ycn.transpose(0, 2, 1))

        Cyy = np.einsum("chs,cds->chd", Yc64, Yc64, optimize=True)  # (C, 2H, 2H)
        self._Wy = _inv_sqrt_psd(Cyy).astype(_DTYPE)

        self._fb_sos_fast = [sos.astype(_DTYPE) for sos in self.fb_sos]

        nyq = self.srate / 2.0
        self._butter_ba = []
        for k in range(1, self.n_subbands + 1):
            low = max(0.1, 6.0 * k)
            high = min(90.0, nyq - 0.1)
            if low >= high:
                self._butter_ba.append(None)  
            else:
                b, a = butter(4, [low / nyq, high / nyq], btype="band")
                self._butter_ba.append((b.astype(_DTYPE), a.astype(_DTYPE)))

    def _score_batch(self, Xb):
        """对一个批次 (T, n_ch, Ns) 计算四路融合得分，返回 (T, C)。"""
        T = len(Xb)
        C = len(self.freqs)
        r = np.zeros((T, C), dtype=_DTYPE)
        Xsubs = self._fast_fb(Xb)

        if self.calibration_status_ == "per_class":
            r += self._rho1_batch(Xsubs)
        r += self._rho23_batch(Xsubs, self._WA)
        r += self._rho23_batch(Xsubs, self._WB)
        r += self._rho4_batch(Xb)
        return r

    def _rho1_batch(self, Xsubs):
        out = None
        for k, Xk in enumerate(Xsubs):
            x = np.matmul(self._W1, Xk)  # (T, C, Ns)
            x -= x.mean(axis=2, keepdims=True)  # 原位中心化
            nx = np.linalg.norm(x, axis=2) + 1e-12  # (T, C)
            num = np.matmul(x.transpose(1, 0, 2), self._YcnT)  # (C, T, 2H)
            cc = np.abs(num).max(axis=2).T  # (T, C)
            term = cc / nx
            wk = float(self.wfb[k])  
            out = term * wk if out is None else out + term * wk
        return out

    def _rho23_batch(self, Xsubs, W):
        out = None
        for k, Xk in enumerate(Xsubs):
            Xd = Xk[..., :: self.tpl_ds]
            y = np.matmul(W, Xd)  # (T, C, S/ds)
            y -= y.mean(axis=2, keepdims=True)  # 原位中心化
            ny = np.linalg.norm(y, axis=2) + 1e-12
            # 模板行已归一化：corr = <y, TplN> / ||y||
            term = np.einsum("tcs,cs->tc", y, self._TplN, optimize=True) / ny
            wk = float(self.wfb[k])
            out = term * wk if out is None else out + term * wk
        return out

    def _rho4_batch(self, Xraw):
        T = len(Xraw)
        C = len(self.freqs)
        n_h = self._Yc.shape[1]  # 2H
        out = np.zeros((T, C), dtype=_DTYPE)
        Yc_flat = self._Yc.reshape(C * n_h, self.n_samples_)  # (C*2H, Ns)
        for k, ba in enumerate(self._butter_ba):
            if ba is None:
                Xk = Xraw.astype(_DTYPE, copy=True)
            else:
                Xk = filtfilt(ba[0], ba[1], Xraw, axis=-1)
            Xk = Xk - Xk.mean(axis=2, keepdims=True)
            Cxx = np.einsum("tcs,tds->tcd", Xk, Xk, optimize=True)  # (T, n_ch, n_ch)
            Wx = _inv_sqrt_psd(Cxx)
            # 互协方差：单次批量 BLAS (T*n_ch, Ns) @ (Ns, C*2H) -> (T, C, n_ch, 2H)
            Cxy = np.matmul(Xk.reshape(T * self.n_ch_, self.n_samples_), Yc_flat.T)
            Cxy = Cxy.reshape(T, self.n_ch_, C, n_h).transpose(0, 2, 1, 3)
            G = np.matmul(Wx[:, None, :, :], Cxy)       # (T, C, n_ch, 2H)
            G = np.matmul(G, self._Wy[None, :, :, :])
            M = np.matmul(np.swapaxes(G, -1, -2), G)    # (T, C, 2H, 2H)
            lam = np.linalg.eigvalsh(M)[..., -1]
            rho = np.sqrt(np.clip(lam, 0.0, 1.0))
            out += float(self.wfb[k]) * rho
        return np.sign(out) * (out ** 2)

    def predict(self, X_test, return_scores=False):
        X = np.asarray(X_test, dtype=_DTYPE)
        if X.ndim == 2:
            X = X[None, ...]
        T = len(X)
        if T == 0:
            pred = np.zeros(0, dtype=int)
            return (pred, np.zeros((0, len(self.freqs)))) if return_scores else pred

        scores = np.empty((T, len(self.freqs)), dtype=_DTYPE)
        for i0 in range(0, T, _CHUNK):
            i1 = min(i0 + _CHUNK, T)
            scores[i0:i1] = self._score_batch(X[i0:i1])
        pred = np.argmax(scores, axis=1)
        if return_scores:
            return pred, scores
        return pred
