"""Recurrent spiking network with stochastic, resource-limited synapses.

Model (Methods, "Spiking network and synapse model"):
  * fixed recurrent layer of leaky integrate-and-fire (LIF) neurons;
  * plastic read-out synapses with calcium-dependent Bernoulli vesicle release
    and a depleting releasable pool;
  * eligibility = low-pass filtered release events (exact e-prop eligibility
    for read-out synapses); learning signal = output error (three-factor rule);
  * synapses are grouped in dendritic branches; one binary decision per branch
    gates whether the branch's candidate update is committed;
  * an ATP-like pool sets the per-window plasticity budget.

All energies are dimensionless proxy units. Nothing here is a calibrated
biophysical ATP model.
"""
from dataclasses import dataclass, field
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


@dataclass
class NetConfig:
    n_in: int = 60
    n_hid: int = 96
    n_groups: int = 12          # dendritic branches = binary decision variables
    T: int = 40                 # time steps per sample
    alpha_m: float = 0.85       # membrane leak per step
    v_th: float = 1.0
    rec_density: float = 0.10
    rec_gain: float = 0.35
    in_gain: float = 0.9
    # synapse
    theta0_mean: float = 0.4    # release-probability logit offset
    theta0_sd: float = 0.6
    theta_C: float = 0.8        # calcium facilitation
    theta_A: float = 0.5        # ATP dependence (held at rest for signalling)
    K_A: float = 5.0
    A_rest: float = 10.0
    alpha_C: float = 0.80
    kappa_C: float = 0.50
    tau_R: float = 8.0          # vesicle recovery time constant (steps)
    nu_R: float = 0.35          # depletion per release
    lam_e: float = 0.92         # eligibility decay
    feat_gain: float = 2.0      # homeostatic feature gain
    # plasticity budget (ATP-like pool, proxy units)
    A_max: float = 12.0
    atp_refill: float = 6.0
    cost_low: int = 1
    cost_high: int = 3


class Network:
    def __init__(self, cfg: NetConfig, rng):
        self.cfg = cfg
        c = cfg
        self.W_in = rng.normal(0, 1, (c.n_hid, c.n_in)) * (rng.random((c.n_hid, c.n_in)) < 0.25) * c.in_gain
        W = rng.normal(0, 1, (c.n_hid, c.n_hid)) * (rng.random((c.n_hid, c.n_hid)) < c.rec_density)
        np.fill_diagonal(W, 0.0)
        self.W_rec = W * c.rec_gain
        self.theta0 = rng.normal(c.theta0_mean, c.theta0_sd, c.n_hid)
        self.group = np.repeat(np.arange(c.n_groups), int(np.ceil(c.n_hid / c.n_groups)))[:c.n_hid]
        self.G = np.zeros((c.n_groups, c.n_hid))
        self.G[self.group, np.arange(c.n_hid)] = 1.0
        self.cost = rng.integers(c.cost_low, c.cost_high + 1, c.n_groups).astype(int)
        # homeostatic baseline: centre/scale eligibility with statistics from random input
        self.mu = np.zeros(c.n_hid); self.sd = np.ones(c.n_hid)
        cal = (rng.random((200, c.T, c.n_in)) < 0.10).astype(float)
        e = self.release(self.hidden_spikes(cal), rng)[0]
        self.mu = e.mean(0); self.sd = e.std(0) + 0.05

    # ---------------------------------------------------------------- dynamics
    def hidden_spikes(self, inp):
        """inp: [batch, T, n_in] binary -> hidden spikes [batch, T, n_hid]."""
        c = self.cfg
        Bt = inp.shape[0]
        v = np.zeros((Bt, c.n_hid))
        z = np.zeros((Bt, c.n_hid))
        out = np.zeros((Bt, c.T, c.n_hid))
        for t in range(c.T):
            I = inp[:, t] @ self.W_in.T + z @ self.W_rec.T
            v = c.alpha_m * v + I - z * c.v_th          # reset by subtraction
            z = (v >= c.v_th).astype(float)
            out[:, t] = z
        return out

    def release(self, spikes, rng):
        """Calcium-dependent Bernoulli release with vesicle depletion.

        Returns eligibility features x [batch, n_hid], mean release probability
        per synapse [n_hid], number of release events, number of spikes."""
        c = self.cfg
        Bt = spikes.shape[0]
        C = np.zeros((Bt, c.n_hid))
        R = np.ones((Bt, c.n_hid))
        e = np.zeros((Bt, c.n_hid))
        atp_term = c.theta_A * c.A_rest / (c.A_rest + c.K_A)
        p_sum = np.zeros(c.n_hid); n_sp = np.zeros(c.n_hid)
        n_rel = 0.0
        for t in range(c.T):
            s = spikes[:, t]
            p = sigmoid(self.theta0 + c.theta_C * C + atp_term)
            avail = rng.random(R.shape) < R
            r = s * avail * (rng.random(p.shape) < p)
            p_sum += (p * s).sum(0); n_sp += s.sum(0)
            n_rel += r.sum()
            e = c.lam_e * e + r
            C = c.alpha_C * C + c.kappa_C * s
            R = np.clip(R + (1.0 - R) / c.tau_R - c.nu_R * r, 0.0, 1.0)
        x = c.feat_gain * (e / np.sqrt(c.T) - self.mu) / (self.sd * np.sqrt(c.n_hid))
        p_mean = np.where(n_sp > 0, p_sum / np.maximum(n_sp, 1), sigmoid(self.theta0 + atp_term))
        return x, p_mean, float(n_rel), float(spikes.sum())


# -------------------------------------------------------------------- tasks
def make_task(cfg: NetConfig, rng, n_active=7):
    """A binary spatio-temporal spike-pattern task (synthetic)."""
    chans = [rng.choice(cfg.n_in, n_active, replace=False) for _ in range(2)]
    early = [bool(rng.integers(0, 2)) for _ in range(2)]
    return dict(chans=chans, early=early)


def sample_task(task, cfg: NetConfig, n, rng, base=0.04, hi=0.30, jitter=0.05):
    y = rng.integers(0, 2, n)
    rates = np.full((n, cfg.T, cfg.n_in), base)
    half = cfg.T // 2
    for cl in (0, 1):
        idx = np.flatnonzero(y == cl)
        sl = slice(0, half) if task["early"][cl] else slice(half, cfg.T)
        for ch in task["chans"][cl]:
            rates[idx, sl, ch] = hi
    rates = np.clip(rates + rng.normal(0, jitter, (n, 1, cfg.n_in)), 0.0, 1.0)
    return (rng.random(rates.shape) < rates).astype(float), y


def encode(net: Network, task, n, rng):
    inp, y = sample_task(task, net.cfg, n, rng)
    x, p_mean, n_rel, n_sp = net.release(net.hidden_spikes(inp), rng)
    return x, y, p_mean, n_rel, n_sp


# ---------------------------------------------------------------- learning
@dataclass
class Readout:
    w: np.ndarray
    b: float = 0.0
    fisher: np.ndarray = None
    anchor: np.ndarray = None

    def prob(self, x):
        return sigmoid(x @ self.w + self.b)

    def accuracy(self, x, y):
        return float(np.mean((self.prob(x) >= 0.5) == y))

    def nll(self, x, y):
        a = x @ self.w + self.b
        return float(np.mean(np.logaddexp(0, a) - y * a))


def window_quadratic(net: Network, ro: Readout, x, y, p_mean, eta=1.0, lam_F=40.0,
                     lam_E=2e-4, lam_U=0.5, use_pair=True):
    """Branch-level QUBO coefficients for one learning window (Methods eqs).

    Returns delta [n_hid], a [n_groups], J [n_groups, n_groups] (upper)."""
    pi = ro.prob(x)
    err = pi - y                                   # learning signal m
    g = (err[:, None] * x).mean(0)                 # three-factor gradient
    delta = -eta * g
    lam = pi * (1 - pi)
    D = net.G * delta[None, :]                     # [groups, n_hid] branch updates
    XD = x @ D.T                                   # [batch, groups]
    Hq = (XD * lam[:, None]).T @ XD / len(y)       # delta_k^T H delta_l
    a = D @ g + 0.5 * np.diag(Hq)
    if ro.fisher is not None:
        a = a + 0.5 * lam_F * net.G @ (ro.fisher * (2 * (ro.w - ro.anchor) * delta + delta ** 2))
    a = a + lam_E * net.cost + lam_U * net.G @ (delta ** 2 * (1 - p_mean))
    J = np.triu(Hq, 1) if use_pair else np.zeros_like(Hq)
    return delta, a, J, float(-eta * err.mean())
