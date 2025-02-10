import io
import sys
import time
import json
import warnings
import contextlib
from pathlib import Path

import pandas as pd
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter

# 初始化DP模型
from deepmd.calculator import DP
calc = DP("graph.pb")

def run_opt(name, pstress, model_name, mlp_trajname, fmax, mlp_optstep):
    verbose = True
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        print("Start to Optimize Structures using ", model_name)
        start = time.time()

        # 读取xyz结构文件
        atoms = read(name)  

        dis_mtx = atoms.get_all_distances(mic=True)
        row, col = np.diag_indices_from(dis_mtx)
        dis_mtx[row, col] = np.nan
        min_dis = np.nanmin(dis_mtx)

        if min_dis > 0.6:
            atoms.calc = calc
            aim_stress = pstress * 0.006242  # 转换应力单位
            ucf = UnitCellFilter(atoms, scalar_pressure=aim_stress)

            opt = FIRE(ucf, trajectory=mlp_trajname)
            opt.run(fmax=fmax, steps=mlp_optstep)

            # 优化后的结果处理
            energy = float(atoms.get_potential_energy())

            # 保存为xyz格式
            target = name.with_suffix('.optdone.vasp')
            write(target, atoms, format='vasp')

            print(f"Optimization done! Cost {time.time() - start:.2f} seconds.")
            
            return {"name": name.stem, "energy": energy}
        else:
            warnings.warn(f"The minimum distance of two atoms is {min_dis}, too close.")
            return None

if __name__ == "__main__":
    results = []
    for name in sorted(Path().glob("*.vasp")):  # 修改为处理xyz文件
        print(f"-------------------Optimizing {name} with DP--------------")
        result = run_opt(
            name,
            pstress=0,  # 压力单位为GPa
            model_name="DP",
            mlp_trajname="traj.traj",
            fmax=0.01,
            mlp_optstep=10000,
        )
        if result:
            results.append(result)

    # 汇总结果并按能量排序
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='energy')
    
    # 保存到CSV文件
    df_sorted.to_csv("sorted_energies.csv", columns=['name', 'energy'], index=False)
    print("Energy sorted summary saved to sorted_energies.csv.")

