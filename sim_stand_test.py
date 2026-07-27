"""CPU MuJoCo stand test for the KUAFU baseline LQR (no jax/MJX needed).

Replicates the sim baseline (_wheel_torque in kuafu_mjx_env.py) and the
firmware-equivalent variant (VX brake, yaw clamp, x_int clamp, pitch-rate
low-pass, pitch fade) in the same harness, then reports stand quality:
pitch excursion, torque chatter (jitter), yaw drift (spin), position drift.

Run:  python sim_stand_test.py [seconds]
"""

import sys
import numpy as np
import mujoco

import kuafu_physics as P

XML = "rl/kuafu.xml"
BASE_DT = P.BASE_DT          # 4 ms control
K = P.LQR_K_DT4              # [K0 K1 K2 K3]
KI = P.LQI_KI_DT4
R = P.R_WHEEL
TAU_STALL = P.TAU_WHEEL_STALL
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * np.pi / 60.0
YAW_KP, YAW_KD = P.YAW_KP, P.YAW_KD
PITCH_FADE_START = 0.349066
PITCH_FADE_END = 0.872665
VX_SAFE_LIMIT = 0.5          # firmware-only brake (m/s)
YAW_CLAMP = 0.1              # firmware-only yaw torque clamp (N m)
XINT_CLAMP = 0.05            # firmware-only integral clamp (m s)
PR_ALPHA = 0.2               # firmware pitch-rate low-pass (1 kHz)


def quat_pitch(q):
    qw, qx, qy, qz = q
    return np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))


def quat_yaw(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz ** 2 + qy ** 2))


def quat_roll(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))


def body_rates(model, data):
    """Local-frame angular velocity (gyro).

    CPU MuJoCo freejoint stores angular velocity already in the body-local
    frame (unlike MJX, which needs the conjugate rotation the env applies)."""
    return data.qvel[3:6]


def limit_wheel_torque(tau, omega):
    available = TAU_STALL * (1.0 - abs(omega) / OMEGA_NOLOAD)
    available = float(np.clip(available, 0.0, TAU_STALL))
    return float(np.clip(tau, -available, available))


def run(variant: str, seconds: float):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    phys_dt = model.opt.timestep
    assert abs(phys_dt - P.PHYS_DT) < 1e-9, f"xml dt {phys_dt}"

    act_tau_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_l")
    act_tau_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_r")
    j_wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_l")
    j_wr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_r")
    wl_adr = model.jnt_dofadr[j_wl]
    wr_adr = model.jnt_dofadr[j_wr]

    steps_per_base = round(BASE_DT / phys_dt)
    steps_per_1k = max(1, round(0.001 / phys_dt))  # 500 Hz physics: fusion every other step

    # controller state
    x_est = x_ref = x_int = v_ref = v_acc = yaw_ref = w_ref = w_acc = 0.0
    pr_filt = 0.0
    pitch = pitch_rate = 0.0

    log = []
    total_steps = int(seconds / phys_dt)
    for step in range(total_steps):
        t = step * phys_dt

        # 1 kHz fusion path (firmware DRDY cadence)
        if step % steps_per_1k == 0:
            q = data.qpos[3:7].copy()
            rates = body_rates(model, data)
            pitch = quat_pitch(q)
            raw_pr = rates[1]
            if variant == "firmware":
                pr_filt += PR_ALPHA * (raw_pr - pr_filt)
                pitch_rate = pr_filt
            else:
                pitch_rate = raw_pr

        # 250 Hz control
        if step % steps_per_base == 0:
            q = data.qpos[3:7].copy()
            rates = body_rates(model, data)
            yaw_rate = rates[2]
            wl = data.qvel[wl_adr]
            wr = data.qvel[wr_adr]
            vx = 0.5 * (wl + wr) * R
            x_est += vx * BASE_DT

            # jerk-limited refs (zero command)
            def jerk(target, value, acc, max_a, max_j):
                a_t = np.clip((target - value) / BASE_DT, -max_a, max_a)
                acc += float(np.clip(a_t - acc, -max_j * BASE_DT, max_j * BASE_DT))
                value += acc * BASE_DT
                return value, acc
            v_ref, v_acc = jerk(0.0, v_ref, v_acc, 2.0, 8.0)
            w_ref, w_acc = jerk(0.0, w_ref, w_acc, 4.0, 16.0)
            x_ref += v_ref * BASE_DT
            yaw_ref = (yaw_ref + w_ref * BASE_DT + np.pi) % (2 * np.pi) - np.pi

            x_error = x_est - x_ref
            if variant == "firmware":
                if abs(pitch) > PITCH_FADE_END:
                    x_int = 0.0
                else:
                    x_int = float(np.clip(x_int + x_error * BASE_DT, -XINT_CLAMP, XINT_CLAMP))
            else:
                x_int += x_error * BASE_DT

            state = np.array([x_error, pitch, vx - v_ref, pitch_rate])
            force = -(K @ state) - KI * x_int
            if variant == "firmware":
                if vx > VX_SAFE_LIMIT:
                    force -= 50.0 * (vx - VX_SAFE_LIMIT)
                elif vx < -VX_SAFE_LIMIT:
                    force -= 50.0 * (vx + VX_SAFE_LIMIT)
            tau_pitch = force * R * 0.5

            yaw = quat_yaw(q)
            yaw_error = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
            tau_yaw = YAW_KP * yaw_error + YAW_KD * (w_ref - yaw_rate)
            if variant == "firmware":
                tau_yaw = float(np.clip(tau_yaw, -YAW_CLAMP, YAW_CLAMP))

            # contract mapping (sim frame): l = pitch - yaw, r = pitch + yaw
            tau_l = float(np.clip(tau_pitch - tau_yaw, -TAU_STALL, TAU_STALL))
            tau_r = float(np.clip(tau_pitch + tau_yaw, -TAU_STALL, TAU_STALL))
            tau_l = limit_wheel_torque(tau_l, wl)
            tau_r = limit_wheel_torque(tau_r, wr)

            if variant == "firmware":
                ap = abs(pitch)
                scale = 1.0 if ap <= PITCH_FADE_START else (
                    0.0 if ap >= PITCH_FADE_END else
                    (PITCH_FADE_END - ap) / (PITCH_FADE_END - PITCH_FADE_START))
                tau_l *= scale
                tau_r *= scale

            data.ctrl[act_tau_l] = tau_l
            data.ctrl[act_tau_r] = tau_r

            log.append((t, pitch, pitch_rate, vx, quat_yaw(q), tau_l, tau_r,
                        data.qpos[0], quat_roll(q)))

        mujoco.mj_step(model, data)

    log = np.array(log)
    t = log[:, 0]
    pitch_l = log[:, 1]
    yaw_l = log[:, 4]
    tau = log[:, 5:7]
    xpos = log[:, 7]
    survived = bool(np.max(np.abs(pitch_l)) < 0.5)
    jitter = float(np.sqrt(np.mean(np.diff(tau, axis=0) ** 2)))  # torque step chatter
    print(f"[{variant:9s}] survived={survived}  max|pitch|={np.max(np.abs(pitch_l)):.4f} rad  "
          f"final pitch={pitch_l[-1]:+.4f}  yaw drift={yaw_l[-1] - yaw_l[0]:+.4f} rad  "
          f"pos drift={xpos[-1] - xpos[0]:+.3f} m  tau rms={np.sqrt((tau ** 2).mean()):.4f}  "
          f"chatter={jitter:.4f} N m/step")
    return log


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run("sim", secs)
    run("firmware", secs)
