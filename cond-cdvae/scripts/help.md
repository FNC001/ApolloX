# generate

1. 生成结构
```
python evaluate.py --model_path <m> --formula <f> --pressure <P(GPa)> --label <label>
# 输出
>>> eval_gen_<label>.pt
```

2. 提取结构
```
python extract_gen.py eval_gen_<label>.pt
# 输出
>>> eval_gen_<label>/gen/*.vasp
>>> eval_gen_<label>/gen.pkl
```

3. 寻找对称性
```
python find_spg.py eval_gen_<label>/gen -s 0.01 -s 0.1 -s 0.5 -s 1.0
# 输出
>>> eval_gen_<label>/std_<symprec>/*.vasp  # 标准化结构
>>> eval_gen_<label>/spg.txt
```

4. 准备VASP计算
```
python prepare_vasp.py eval_gen_<label>/std_<symprec> -s <nsw> -p <P(kbar)>
# 输出
>>> eval_gen_<label>/std_<symprec>.scf/*/...  # NSW<=1
>>> eval_gen_<label>/std_<symprec>.opt/*/...  # NSW>1  ISYM=0 统一比较优化时间/步数
```

5. 准备对应CALYPSO input.dat
```
python prepare_calypso.py eval_gen_<label>/std_<symprec>.<std/opt> -r 0.7
# 输出
>>> eval_gen_<label>/std_<symprec>.<scf/opt>/*/caly/input.dat
```
