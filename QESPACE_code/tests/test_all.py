"""Unit and consistency tests:  python -m pytest -q"""
import numpy as np
from qespace import qubo as Q, qaoa, noise as NZ, iceberg as IB, reservoir as RS


def _inst(n=6, seed=0):
    r = np.random.default_rng(seed)
    a = r.normal(size=n); J = np.triu(r.normal(size=(n, n)) * 0.3, 1)
    c = r.integers(1, 4, n); B = int(c.sum() // 2)
    return a, J, c, B


def test_encodings_are_exact():
    a, J, c, B = _inst()
    f = Q.all_values(a, J); bits = Q.all_bits(len(a)); hslack = B - bits @ c
    a2, J2, k0 = Q.unbalanced_qubo(a, J, c, B, 0.7, 0.3)
    assert np.allclose(Q.all_values(a2, J2) + k0, f - 0.7 * hslack + 0.3 * hslack ** 2)
    a3, J3, k1 = Q.slack_qubo(a, J, c, B, 2.0)
    full = Q.all_bits(len(a3)); nb = len(a3) - len(a)
    pen = 2.0 * (full[:, :len(a)] @ c + full[:, len(a):] @ (2.0 ** np.arange(nb)) - B) ** 2
    fu = full[:, :len(a)] @ a + np.einsum("bi,ij,bj->b", full[:, :len(a)], J, full[:, :len(a)])
    assert np.allclose(Q.all_values(a3, J3) + k1, fu + pen)


def test_ising_map_and_scale():
    a, J, c, B = _inst(7, 1)
    h, K, const = Q.qubo_to_ising(a, J)
    z = 1 - 2 * Q.all_bits(7)
    E = z @ h + np.einsum("bi,ij,bj->b", z, K, z) + const
    assert np.allclose(E, Q.all_values(a, J))
    assert np.isclose(np.std(E), Q.ising_scale(h, K), rtol=1e-9)


def test_classical_solvers():
    a, J, c, B = _inst(10, 2)
    u, v = Q.exact_feasible(a, J, c, B)
    for fn in (Q.greedy_feasible, Q.knapsack_diag):
        uu, vv = fn(a, J, c, B); assert c @ uu <= B and vv >= v - 1e-12
    uu, vv = Q.simulated_annealing(a, J, c, B, np.random.default_rng(0)); assert c @ uu <= B and vv >= v - 1e-12
    ud, vd = Q.knapsack_diag(a, 0 * J, c, B); ue, ve = Q.exact_feasible(a, 0 * J, c, B)
    assert np.isclose(vd, ve)


def test_gate_simulator_matches_statevector_qaoa():
    a, J, c, B = _inst(6, 3)
    h, K, _ = Q.qubo_to_ising(a, J); s = Q.ising_scale(h, K)
    g, b = [0.7, 1.3], [0.5, 0.2]
    sim = NZ.Sim(6); NZ.qaoa_circuit(sim, h / s, K / s, g, b)
    assert np.allclose(sim.probs(), qaoa.qaoa_probs(qaoa.normalised_energies(a, J), g, b), atol=1e-10)


def test_qaoa_beats_uniform():
    a, J, c, B = _inst(8, 4)
    E = qaoa.normalised_energies(a, J)
    x, f = qaoa.train_angles([E], 1, restarts=1, maxiter=150)
    assert f < -0.2


def test_iceberg_noiseless_equals_unencoded_and_detects_errors():
    a, J, c, B = _inst(6, 5)
    h, K, _ = Q.qubo_to_ising(a, J); s = Q.ising_scale(h, K); h, K = h / s, K / s
    g, b = [0.6], [0.4]
    ref = NZ.Sim(6); NZ.qaoa_circuit(ref, h, K, g, b)
    enc = NZ.Sim(8); IB.iceberg_qaoa(enc, h, K, g, b)
    bits = Q.all_bits(8).astype(int); logical, ok = IB.decode(bits)
    p = enc.probs()
    assert np.isclose(p[ok].sum(), 1.0, atol=1e-10)             # code space only
    marg = np.zeros(64); np.add.at(marg, logical @ (1 << np.arange(6)), p[ok])
    assert np.allclose(marg, ref.probs(), atol=1e-10)
    rng = np.random.default_rng(0)
    pn, _ = NZ.averaged_probs(lambda s_: IB.iceberg_qaoa(s_, h, K, g, b), 8, NZ.NoiseModel(p2=0.02), rng, 60)
    assert pn[~ok].sum() > 0.01                                  # errors leave the code space


def test_swap_network_touches_every_pair_once():
    for n in (5, 8, 12):
        layers, perm = Q.swap_network_schedule(n)
        seen = sorted(tuple(sorted((u, v))) for row in layers for (_, u, v) in row)
        assert seen == [(i, j) for i in range(n) for j in range(i + 1, n)]
        assert perm == list(range(n))[::-1]


def test_readout_mitigation_and_zne():
    rng = np.random.default_rng(1); n = 5
    h = rng.normal(size=n); K = np.triu(rng.normal(size=(n, n)), 1)
    p = rng.dirichlet(np.ones(1 << n))
    z = 1 - 2 * Q.all_bits(n); exact = float(p @ (z @ h + np.einsum("bi,ij,bj->b", z, K, z)))
    nm = NZ.NoiseModel(p01=0.03, p10=0.06)
    bits = NZ.sample_with_readout(p, 400000, nm, rng)
    raw = NZ.ising_expectation(bits, h, K); mit = NZ.ising_expectation(bits, h, K, (0.03, 0.06))
    assert abs(mit - exact) < abs(raw - exact) and abs(mit - exact) < 0.02
    lam = np.array([1, 3, 5.0])
    assert np.isclose(NZ.extrapolate(lam, -2.0 * np.exp(-0.1 * lam), "exponential"), -2.0)
    assert np.isclose(NZ.extrapolate(lam, 1 - 0.1 * lam, "linear"), 1.0)


def test_reservoir_is_physical_and_has_memory():
    rng = np.random.default_rng(0); s = rng.uniform(0, 1, 300)
    F = RS.quantum_features(s[:, None], n_res=2)
    assert F.shape == (300, 6) and np.abs(F).max() <= 1 + 1e-9
    y = np.r_[0, s[:-1]]
    assert RS.ridge_eval(F, y, 20, 200)[1] > 0.1                 # remembers s_{t-1}
    Ft = RS.CollisionReservoir(trotter=True).run(s[:50, None]); assert np.isfinite(Ft).all()


def test_learning_loop_respects_budget():
    from qespace.snn import NetConfig
    from experiments.exp1_continual import run_seed
    out = run_seed(0, NetConfig(n_hid=48, n_groups=6), n_tasks=2, windows=6, n_test=40,
                   allocators=["dense", "greedy", "exact"])
    assert out["exact"]["update_cost"] < out["dense"]["update_cost"]
    assert out["exact"]["hit_opt"] == 1.0
