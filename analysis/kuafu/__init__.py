# -*- coding: utf-8 -*-
"""
KUAFU 分析包
  mechanism — 机构解算 (运动学/静力学/动力学/力椭球几何), 纯 numpy
  styling   — 统一视觉系统 (色盲安全色板 + 出版级排版 + 绘图工具)
"""
from .mechanism import (
    # 常量
    G, MM, AX, BX, A, B, A_LEN, B_LEN, XQ, R_WHEEL,
    M_TOT, F_GRAV, F_DES, TAU_STALL, TAU_CONT, W_SERVO,
    M_CRANK, M_LINK, M_WHEEL,
    # 运动学
    solve_chain, forward_kin, kin, jacobian, reachable, ik_full,
    # 参数化运动学 (供蒙特卡洛优化)
    _hip_points, kin_param, forward_kin_param, jacobian_param,
    # 动力学
    leg_points, kinetic_energy, mass_matrix, gravity_vec,
    # 力椭球
    torque_ellipse, force_ellipse,
)
from .styling import (
    C, OI, PALETTE, SAFE, CAUTION, DANGER, ACCENT, HIGHLIGHT, NEUTRAL,
    SERIES, D0_COLORS,
    OUTPUT, savefig, ax_clean, add_zone, annotate_point,
    set_title_style, final_style,
)
