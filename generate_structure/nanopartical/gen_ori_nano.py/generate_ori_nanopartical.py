from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np

# 创建一个面心立方（fcc）结构的铜单晶体
cu_bulk = bulk('Cu', 'fcc', a=3.6)  # a是铜的晶格常数

# 生成纳米颗粒，通过切割一个大的晶体来模拟
def create_nanoparticle(atom, size, vacuum=10.0):
    # 通过复制原始晶体来增大尺寸
    supercell = atom * (size, size, size)
    
    # 计算颗粒的半径（假设是球形的）
    radius = size * atom.cell.lengths()[0] / 2
    
    # 创建一个掩码，用于选择位于球形区域内的原子
    center = np.sum(supercell.positions, axis=0) / len(supercell)
    mask = np.linalg.norm(supercell.positions - center, axis=1) < radius
    
    # 使用掩码筛选原子
    nanoparticle_atoms = supercell[mask]

    # 计算包围盒大小并添加真空层
    cell_size = np.ptp(nanoparticle_atoms.positions, axis=0) + 2 * vacuum
    nanoparticle = Atoms(nanoparticle_atoms.symbols, positions=nanoparticle_atoms.positions - nanoparticle_atoms.positions.min(axis=0) + vacuum, cell=cell_size, pbc=True)

    return nanoparticle

# 设置颗粒的尺寸（单位是原子数，沿每个方向）
nanoparticle_size = 8  # 例如5x5x5个原子
vacuum_layer = 15.0    # 添加15埃的真空层

try:
    nanoparticle = create_nanoparticle(cu_bulk, nanoparticle_size, vacuum_layer)
    # 保存为POSCAR格式
    write('POSCAR', nanoparticle, format='vasp')
    print("POSCAR file has been saved with the nanoparticle including a vacuum layer.")
except Exception as e:
    print("An error occurred while saving the POSCAR file:", e)

from ase.visualize import view
view(nanoparticle)
