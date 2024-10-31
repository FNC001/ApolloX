#!/usr/bin/env python
import io
import sys
import time
import warnings
import contextlib
from pathlib import Path

import pandas as pd
import numpy as np
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md import MDLogger
from ase import units

from chgnet.model import CHGNet
from chgnet.model import CHGNetCalculator as MLP

# 初始化CHGNet模型
chgnet = CHGNet.load()
calc = MLP(chgnet, use_device="cuda:0")  # 使用GPU

def run_md(name, temp_start, temp_end, timestep, steps, md_trajname, loginterval):
    verbose = True
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        print(f"开始对 {name} 进行分子动力学模拟，使用 CHGNET")
        start = time.time()

        atoms = read(name)  # 读取结构文件

        # 设置CHGNet计算器
        atoms.calc = calc

        # 设置初始速度分布
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_start)

        # 创建分子动力学模拟器
        dyn = VelocityVerlet(atoms, timestep * units.fs, trajectory=md_trajname)

        # 打开日志文件
        logfile = open('md.log', 'a')
        logger = MDLogger(dyn, atoms, logfile, header=True, peratom=False, mode="a")

        dyn.attach(logger, interval=loginterval)

        # 定义温度调度函数，实现线性升温或降温
        def scale_temperature():
            current_step = dyn.nsteps
            total_steps = steps
            current_temp = temp_start + (temp_end - temp_start) * (current_step / total_steps)
            # 重新缩放速度以达到当前温度
            kinetic_energy = atoms.get_kinetic_energy()
            current_temp_actual = kinetic_energy / (1.5 * units.kB * len(atoms))
            scale_factor = (current_temp / current_temp_actual) ** 0.5
            atoms.set_velocities(atoms.get_velocities() * scale_factor)

        # 定义状态输出函数，输出能量、受力、温度等信息
        def print_status():
            epot = atoms.get_potential_energy() / len(atoms)
            ekin = atoms.get_kinetic_energy() / len(atoms)
            etot = epot + ekin
            temp = ekin / (1.5 * units.kB)
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            print(f"Step: {dyn.nsteps}, E_pot: {epot:.6f} eV/atom, "
                  f"E_kin: {ekin:.6f} eV/atom, E_tot: {etot:.6f} eV/atom, "
                  f"Temp: {temp:.2f} K, Max force: {max_force:.6f} eV/Å")

        # 每个时间步调用温度调度函数
        dyn.attach(scale_temperature, interval=1)
        # 每隔指定步数输出状态信息
        dyn.attach(print_status, interval=loginterval)

        # 运行分子动力学模拟
        dyn.run(steps)

        logfile.close()
        final_structure = atoms
        final_structure_filename = f"{name.stem}_MDdone.vasp"
        write(final_structure_filename, final_structure, format='vasp')
        print(f"分子动力学模拟完成！耗时 {time.time() - start:.2f} 秒。")

if __name__ == "__main__":
    for name in sorted(Path().glob("POSCAR-*.vasp")):
        print(f"-------------------对 {name} 进行MD模拟，使用 CHGNET--------------")
        run_md(
            name,
            temp_start=300,     # 初始温度（K）
            temp_end=800,       # 终止温度（K）
            timestep=1.0,       # 时间步长（fs）
            steps=1000,         # 总步数
            md_trajname=f"{name.stem}_md.traj",
            loginterval=10      # 日志记录间隔
        )

    print("所有MD模拟已完成。")
