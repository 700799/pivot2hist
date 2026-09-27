"""Hidden Markov models: Baum-Welch (EM) for categorical emissions, no hmmlearn/scipy.

A bare transition matrix (:mod:`._chains`) treats every step of a sequence as coming
from one single process. Often it doesn't: a user's login events look different during
normal use than during a burst of failed attempts, a host's connections look different
during routine traffic than during a scan — an unobserved "regime" switches the emission
and transition behaviour underneath the same observed event stream. An HMM recovers that
regime: a small number of hidden states, each with its own typical mix of events, with
its own transition tendencies between regimes.

:func:`decode_regimes` is the entry point most callers want: fit an HMM to per-entity
event sequences and Viterbi-decode a regime label for every row, ready to use as a pivot
dimension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMFit:
    """A fitted HMM: ``n_states`` hidden regimes emitting from ``symbols``."""

    symbols: Tuple[str, ...]  # emission vocabulary, index-aligned to B's columns
    n_states: int
    pi: np.ndarray  # (K,) initial state distribution
    A: np.ndarray  # (K, K) transition matrix, A[i, j] = P(state j | state i)
    B: np.ndarray  # (K, M) emission matrix, B[i, m] = P(symbol m | state i)
    loglik: float
    bic: float
    n_obs: int  # total observations across every sequence fit

    def viterbi(self, obs: Sequence[int]) -> np.ndarray:
        """Most likely hidden-state path for one integer-coded sequence (log-space)."""
        T = len(obs)
        K = self.n_states
        if T == 0:
            return np.zeros(0, dtype=int)
        logA = np.log(self.A + 1e-300)
        logB = np.log(self.B + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA  # (from, to)
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[:, obs[t]]
        path = np.empty(T, dtype=int)
        path[-1] = int(delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def state_summary(self, top: int = 3) -> List[Dict]:
        """Per regime: its steady-state share and its most typical emissions."""
        stationary = _stationary(self.A)
        out = []
        for i in range(self.n_states):
            order = np.argsort(-self.B[i])[:top]
            out.append({
                "regime": i + 1,
                "share": float(stationary[i]),
                "typical": [{"symbol": self.symbols[j], "prob": float(self.B[i, j])} for j in order],
            })
        return out

    def __repr__(self) -> str:
        return f"<HMMFit states={self.n_states} symbols={len(self.symbols)} bic={self.bic:.1f}>"


def _stationary(A: np.ndarray, iters: int = 200) -> np.ndarray:
    k = A.shape[0]
    pi = np.full(k, 1.0 / k)
    for _ in range(iters):
        new = pi @ A
        if np.abs(new - pi).sum() < 1e-12:
            return new
        pi = new
    return pi


def _encode(sequences: Sequence[Sequence[str]]) -> Tuple[List[np.ndarray], Tuple[str, ...]]:
    vocab = sorted({str(s) for seq in sequences for s in seq})
    index = {s: i for i, s in enumerate(vocab)}
    encoded = [np.array([index[str(s)] for s in seq], dtype=int) for seq in sequences]
    return encoded, tuple(vocab)


def fit_hmm(
    sequences: Sequence[Sequence[str]], n_states: int, *, seed: int = 0, iters: int = 60, tol: float = 1e-4, n_init: int = 3
) -> HMMFit:
    """Fit an ``n_states``-regime categorical HMM to ``sequences`` (one list of
    symbols per independent entity) by scaled forward-backward Baum-Welch, best of
    ``n_init`` random restarts."""
    encoded, vocab = _encode(sequences)
    encoded = [o for o in encoded if len(o) > 0]
    if not encoded:
        raise ValueError("need at least one non-empty sequence")
    M = len(vocab)
    K = max(1, min(int(n_states), M if M > 0 else 1))
    rng = np.random.default_rng(seed)
    best: Optional[HMMFit] = None
    for _ in range(max(1, n_init)):
        pi = rng.dirichlet(np.ones(K))
        A = rng.dirichlet(np.ones(K) * 3, size=K)  # mild self-transition bias: regimes should persist a little
        B = rng.dirichlet(np.ones(M), size=K)
        prev_ll = -math.inf
        for _ in range(iters):
            pi_num = np.zeros(K)
            A_num = np.zeros((K, K))
            A_den = np.zeros(K)
            B_num = np.zeros((K, M))
            B_den = np.zeros(K)
            total_ll = 0.0
            for obs in encoded:
                T = len(obs)
                alpha = np.empty((T, K))
                c = np.empty(T)
                alpha[0] = pi * B[:, obs[0]]
                c[0] = alpha[0].sum() + 1e-300
                alpha[0] /= c[0]
                for t in range(1, T):
                    alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
                    c[t] = alpha[t].sum() + 1e-300
                    alpha[t] /= c[t]
                total_ll += float(np.log(c).sum())
                beta = np.empty((T, K))
                beta[T - 1] = 1.0
                for t in range(T - 2, -1, -1):
                    beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
                pi_num += gamma[0]
                if T > 1:
                    for t in range(T - 1):
                        xi_t = (alpha[t][:, None] * A * (B[:, obs[t + 1]] * beta[t + 1])[None, :]) / c[t + 1]
                        A_num += xi_t
                    A_den += gamma[:-1].sum(axis=0)
                for t in range(T):
                    B_num[:, obs[t]] += gamma[t]
                B_den += gamma.sum(axis=0)
            if abs(total_ll - prev_ll) < tol * (abs(prev_ll) + 1):
                prev_ll = total_ll
                break
            prev_ll = total_ll
            pi = pi_num / max(len(encoded), 1)
            pi = pi / pi.sum()
            A = A_num / np.maximum(A_den, 1e-300)[:, None]
            A = A / A.sum(axis=1, keepdims=True)
            B = B_num / np.maximum(B_den, 1e-300)[:, None]
            B = B / B.sum(axis=1, keepdims=True)
        if not np.isfinite(prev_ll):
            continue
        n_obs = sum(len(o) for o in encoded)
        k_params = (K - 1) + K * (K - 1) + K * (M - 1)
        bic = k_params * math.log(max(n_obs, 2)) - 2 * prev_ll
        fit = HMMFit(vocab, K, pi, A, B, prev_ll, bic, n_obs)
        if best is None or fit.bic < best.bic:
            best = fit
    if best is None:  # pragma: no cover - only on pathological all-restart failure
        raise ValueError("HMM fit did not converge on any restart")
    return best


def choose_hmm_states(sequences: Sequence[Sequence[str]], k_max: int = 4, *, seed: int = 0, n_init: int = 2) -> int:
    """Best regime count in ``1..k_max`` by BIC."""
    best_k, best_bic = 1, math.inf
    for k in range(1, max(1, k_max) + 1):
        try:
            fit = fit_hmm(sequences, k, seed=seed, n_init=n_init)
        except ValueError:
            break
        if fit.bic < best_bic - 1e-6:
            best_k, best_bic = k, fit.bic
        elif k > best_k + 1:
            break
    return best_k


def decode_regimes(
    df: pd.DataFrame,
    state: str,
    *,
    by: Optional[Sequence[str]] = None,
    time: Optional[str] = None,
    n_states: Optional[int] = None,
    k_max: int = 4,
    seed: int = 0,
    name: str = "regime",
) -> pd.Series:
    """Fit an HMM to ``state``'s per-``by``-entity sequences and Viterbi-decode a regime
    label for every row, aligned to ``df.index`` (ordered categorical ``"regime 1"``,
    ``"regime 2"`` ... numbered by steady-state share, largest first).
    """
    by_l = [by] if isinstance(by, str) else (list(by) if by else [])
    sort_cols = by_l + ([time] if time else [])
    d = df.sort_values(sort_cols, kind="stable") if sort_cols else df
    groups = d.groupby(by_l, sort=False, observed=True) if by_l else [(None, d)]
    seqs: List[List[str]] = []
    idx_per_seq: List[np.ndarray] = []
    for _, g in groups:
        seqs.append(g[state].astype(str).tolist())
        idx_per_seq.append(g.index.to_numpy())
    non_empty = [s for s in seqs if s]
    if not non_empty:
        raise ValueError(f"no data to decode regimes from in {state!r}")
    k = n_states if n_states else choose_hmm_states(non_empty, k_max=k_max, seed=seed)
    fit = fit_hmm(non_empty, max(1, k), seed=seed)
    vocab_index = {s: i for i, s in enumerate(fit.symbols)}
    # renumber regimes by steady-state share, largest first, for a stable/readable label order
    order = np.argsort(-_stationary(fit.A))
    rank = {int(old): i + 1 for i, old in enumerate(order)}
    labels = [f"regime {i + 1}" for i in range(fit.n_states)]
    out = pd.Series(index=df.index, dtype=object)
    for seq, idxs in zip(seqs, idx_per_seq):
        if len(seq) == 0:
            continue
        obs = np.array([vocab_index[s] for s in seq], dtype=int)
        path = fit.viterbi(obs)
        out.loc[idxs] = [f"regime {rank[int(p)]}" for p in path]
    return pd.Series(pd.Categorical(out, categories=labels, ordered=True), index=df.index, name=name)


__all__ = ["HMMFit", "fit_hmm", "choose_hmm_states", "decode_regimes"]
