"""Native MuJoCo model and current baseline-controller verification."""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import kuafu_physics as P
from rl.verify.verify_baseline_ctl import run


def main() -> int:
    xml = os.path.join(ROOT, "rl", "kuafu.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("model dimensions", model.nq == 17 and model.nv == 16 and model.nu == 6,
                   f"nq={model.nq} nv={model.nv} nu={model.nu}"))
    checks.append(("closed-chain constraints", model.neq == 2, f"neq={model.neq}"))
    checks.append(("physics timestep", abs(model.opt.timestep - P.PHYS_DT) < 1e-9,
                   f"dt={model.opt.timestep}"))
    wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_l_geom")
    checks.append(("MJX-compatible wheel geometry", model.geom_type[wheel_id] == mujoco.mjtGeom.mjGEOM_CAPSULE and
                   abs(model.geom_size[wheel_id, 0] - P.R) < 1e-9 and
                   abs(model.geom_size[wheel_id, 1] - 0.0087) < 1e-9,
                   f"type={model.geom_type[wheel_id]} size={model.geom_size[wheel_id]}"))
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    qa = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "Q_A_l")
    qb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "Q_B_l")
    checks.append(("dwell foot geometry", abs(data.site_xpos[qa][2] - (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")][2] - P.D0_MIN * P.MM)) < 2e-3,
                   f"Q_A={data.site_xpos[qa]}"))
    checks.append(("dwell closure", np.linalg.norm(data.site_xpos[qa] - data.site_xpos[qb]) < 1e-3,
                   f"gap={np.linalg.norm(data.site_xpos[qa] - data.site_xpos[qb]):.6f}m"))
    for name, command in (("hold", (0.0, 0.0)), ("forward", (0.5, 0.0)),
                          ("reverse", (-0.5, 0.0)), ("yaw+", (0.0, 0.3)),
                          ("yaw-", (0.0, -0.3))):
        result = run(*command)
        if name == "hold":
            ok = not result["fallen"] and abs(result["dx"]) < 0.15
        elif name in ("forward", "reverse"):
            ok = not result["fallen"] and np.sign(result["xdot"]) == np.sign(command[0])
        else:
            ok = not result["fallen"] and np.sign(result["yaw_est"]) == np.sign(command[1])
        checks.append((f"baseline {name}", ok, str(result)))

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"verify_model: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
