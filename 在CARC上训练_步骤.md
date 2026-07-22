# 在 CARC 上训练（更大数据 + 更强方法）

## 你要做的两个选择（先定）

**A. 数据用哪份?**
- **加州外海 `ca_offshore.shp`(推荐先用)**:海底断层密(1093 条),和"海底地形"同域,**对症**。配上新的 2D 切分,训练数据比之前多约 70%。
- **GEM 全球库**:覆盖全球但**绝大多数是陆地断层**(加州框里只有 119 条,反而比 ca_offshore 少)。适合做"陆地预训练 / 其他区域",不适合直接当加州海底训练。

> 结论:**先用 ca_offshore 跑这版"更大+更强"**;GEM 留给之后的迁移学习/全球扩展。两份都用同一个脚本 `prepare_dataset_gem.py`(它自动识别两种目录格式)。

**B. 已经帮你做好的"更强方法"**(都已写进 `trainV10.py`):160 轮、自动选 GPU、无泄漏空间切分、`num_workers=0`;加上 k1=k0、`--d0 0.03 --dw 0.14 --alpha 0.02` 的损失重配。

---

## 步骤(在 CARC 上,按顺序)

### 0. 把代码传上 CARC
把整个 `SAM code` 文件夹传到 CARC(本地 Mac 上):
```
scp -r "/Users/.../Earthquake - SAM/SAM_code_extracted/SAM code" \
       "ljin7532@10.72.0.13:/scratch1/ljin7532/sam_fault"
```
(全球 GEBCO 你已经在 CARC 上了;GEM 不用传——下一步直接在 CARC clone。)

### 1. 装环境(一次性,登录节点)
```
cd /scratch1/ljin7532/sam_fault
bash setup_carc.sh
```

### 2. 拿断层目录
```
# GEM(小,几十 MB)
git clone https://github.com/GEMScienceTools/gem-global-active-faults.git
# ca_offshore 已在 Qfaults_GIS/SHP/ 里(随代码传上来的)
```

### 3. 生成"更大"的数据集(登录节点,几分钟)
用 ca_offshore + 全球 GEBCO,框一个比之前更大的加州范围:
```
module load conda gcc && source activate sam_env
python prepare_dataset_gem.py \
  --nc /scratch1/ljin7532/GEBCO_2026.nc \
  --shp Qfaults_GIS/SHP/ca_offshore.shp \
  --bbox -125 31.5 -116.5 41 \
  --out earth_dataset_v2
```
(注意:解压出的全球 nc 文件名可能不是 `GEBCO_2026.nc`,用 `ls *.nc` 确认后替换。)

### 4. 让 trainFile 指向新数据 + 把 k1 设成 k0
```
sed -i "s#trainFile *=.*#trainFile = 'earth_dataset_v2/faults.h5'#g" trainV10.py
# k1=k0(集中火力做"有无断层")
python - <<'EOF'
import h5py
f=h5py.File("earth_dataset_v2/faults.h5","a")
for k in list(f.keys()):
    g=f[k]
    if "k0" in g and "k1" in g:
        k0=g["k0"][:]; del g["k1"]; g.create_dataset("k1",data=k0,compression="gzip")
f.close(); print("k1=k0 done")
EOF
```

### 5. 改 SLURM 脚本里的两处,然后提交
打开 `train_carc.slurm`,改 `--partition`、`--account`(用 `myaccount` 查你的账号),确认 GPU 类型。然后:
```
sbatch --export=ALL,DATASET=earth_dataset_v2,RESDIR=train_v2_carc/ train_carc.slurm
squeue -u ljin7532
```

### 6. 看结果
任务结束后:
```
cat train_v2_carc/loss.txt            # 应持续下降
cat train_v2_carc/test_metrics.json   # AP / F1 / IoU(和之前 0.22 对比)
```
把 `test_metrics.json` 发我,我对比上一版看提升多少。

---

## 提醒
- **GPU 上 batch 可以加大**:Mac 上只能 `--batchsize 2`,CUDA GPU 上脚本里设了 8,显存够还能更大(快很多)。
- **先确认 CARC 细节**:你有没有 `conda` 模块?GPU 分区叫什么(`gpu`?)?账号字符串?这三样填进 `setup_carc.sh` / `train_carc.slurm` 才能跑。不确定的话把 `module avail conda` 和 `sinfo -s` 的输出发我,我帮你填。
- **想用 GEM 跑陆地预训练**:把第 3 步的 `--shp` 换成 `gem-global-active-faults/shapefile/gem_active_faults_harmonized.shp`,`--bbox` 框一块陆地断层密的区域即可,其余步骤一样。
