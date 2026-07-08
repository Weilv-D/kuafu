# -*- coding: utf-8 -*-
"""
KUAFU 仿真模型可视化 — MuJoCo interactive viewer

加载 kuafu.xml 驻留态 keyframe，启动交互 viewer 供肉眼检查姿态/碰撞/闭链。
无 GPU 依赖（原生 MuJoCo，CPU 即可），需图形显示环境。

运行: python rl/verify/launch_viewer.py
快捷键: Space 暂停, P 屏幕截图, F 1~N 切换 keyframe, 双击 body 施力
"""
import os
import sys
import mujoco
import mujoco.viewer

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    # 加载驻留态 keyframe
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"加载驻留态 keyframe: 轮中心 Z={d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wheel_l')][2]:.4f} m")
    print(f"总质量 {m.body_mass.sum():.3f} kg, nq={m.nq}, nu={m.nu}")
    print("启动 viewer（关闭窗口退出）...")
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
