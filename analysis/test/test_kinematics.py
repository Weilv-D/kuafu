# -*- coding: utf-8 -*-
"""KUAFU 核心运动学/静力学/动力学 — 单元测试
运行: pytest test_kinematics.py -v
"""
import numpy as np
import pytest

import kuafu as kc
from kuafu import (solve_chain, kin, forward_kin, jacobian,
                   A, B, A_LEN, B_LEN, F_DES, F_GRAV, TAU_CONT, TAU_STALL, MM)


class TestSolveChain:
    def test_solve_both_chains_at_dwell(self):
        Q = np.array([0, -58])
        assert solve_chain(A, A_LEN, B_LEN, Q, 1) is not None
        assert solve_chain(B, A_LEN, B_LEN, Q, 0) is not None

    def test_solve_out_of_range(self):
        Q = np.array([0, -250])
        assert solve_chain(A, A_LEN, B_LEN, Q, 1) is None


class TestKinematics:
    def test_dwell_torque(self):
        r = kin(58, F_DES)
        assert abs(max(r["tau1"], r["tau2"]) - 0.51) < 0.02

    def test_peak_torque(self):
        r = kin(145, F_DES)
        assert abs(max(r["tau1"], r["tau2"]) - 1.92) < 0.03

    def test_max_torque(self):
        r = kin(207, F_DES)
        assert abs(max(r["tau1"], r["tau2"]) - 1.44) < 0.03

    def test_gamma_above_30(self):
        for d in range(58, 208, 5):
            r = kin(d, F_DES)
            if r: assert r["gamma"] >= 30, f"D0={d} γ={r['gamma']:.0f}°"

    def test_symmetry(self):
        for d in [58, 100, 150, 207]:
            r = kin(d, F_DES)
            ratio = min(r["tau1"], r["tau2"]) / max(r["tau1"], r["tau2"])
            assert ratio > 0.95

    def test_grav_load_safe(self):
        # 静载(真实重量均分两腿)扭矩应远低于连续安全值.
        # 阈值 0.70: M_TOT=2.205kg 时静载峰值 ≈0.66 TAU_CONT, 留余量.
        for d in range(58, 208, 5):
            r = kin(d, F_GRAV)
            if r: assert max(r["tau1"], r["tau2"]) / TAU_CONT <= 0.70

    def test_design_below_stall(self):
        for d in range(58, 208, 5):
            r = kin(d, F_DES)
            if r: assert max(r["tau1"], r["tau2"]) < TAU_STALL


class TestJacobian:
    def test_defined_full_range(self):
        for d in range(58, 208, 5):
            J, r = jacobian(d)
            assert J is not None and r is not None

    def test_condition_below_2(self):
        for d in range(58, 208, 5):
            J, _ = jacobian(d)
            sv = np.linalg.svd(J, compute_uv=False)
            assert sv[0]/sv[1] < 2.0

    def test_forward_kin_consistency(self):
        r = kin(58, F_DES)
        Q = forward_kin(r["al1"], r["al2"])
        assert abs(Q[0]) < 0.5 and abs(Q[1] + 58) < 0.5


class TestWorkspace:
    def test_dwell_reachable_center(self):
        r = kc.reachable(0, -58)
        assert r and r["ok"]

    def test_dwell_outside_xband(self):
        r = kc.reachable(50, -58)   # 超过 ±45mm 半宽
        assert not r["ok"]


class TestForceEllipse:
    def test_L2_norm_constant_on_ellipse(self):
        """力椭球边界上 |J^T F|_2 应恒等于 tau_lim (L2 范数定义)"""
        for d in [58, 100, 150, 207]:
            J, _ = jacobian(d)
            axes, angs, F_ell = kc.force_ellipse(J, tau_lim=1.0)
            tau = J.T @ F_ell              # Nmm
            L2 = np.sqrt(tau[0]**2 + tau[1]**2)
            assert abs(L2.max() - 1000.0) < 1e-6, f"D0={d} L2max={L2.max()}"
            assert abs(L2.min() - 1000.0) < 1e-6, f"D0={d} L2min={L2.min()}"

    def test_torque_ellipse_axes_equal_sigma(self):
        """扭矩椭球半轴应等于 sigma_i*MM"""
        for d in [58, 150, 207]:
            J, _ = jacobian(d)
            S = np.linalg.svd(J, compute_uv=False)
            t1, t2 = kc.torque_ellipse(J, F=1.0)
            r_max = np.max(np.hypot(t1, t2))
            assert abs(r_max - S[0]*MM) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
