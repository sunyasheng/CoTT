# 服务器批量处理命令

## 环境准备

在服务器上，首先确保：
1. Python 环境已配置（包含 openai 包）
2. 环境变量已设置（AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT 等）
3. 脚本已上传到服务器

## 快速开始

### 1. 测试运行（处理前 100 张图片）

```bash
cd /path/to/scripts/aaaj_creator

python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --max-images 100 \
    --batch-size 50
```

### 2. 处理所有图片（推荐使用后台运行）

```bash
cd /path/to/scripts/aaaj_creator

nohup python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --batch-size 100 \
    > batch_dpg_processing.log 2>&1 &
```

### 3. 使用 screen 或 tmux（推荐）

```bash
# 使用 screen
screen -S dpg_processing
cd /path/to/scripts/aaaj_creator

python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --batch-size 100

# 按 Ctrl+A 然后 D 来 detach
# 重新连接: screen -r dpg_processing
```

### 4. 断点续传（如果中途中断）

```bash
# 假设已经处理了 5000 张，从第 5000 张继续
python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --start-idx 5000 \
    --batch-size 100
```

## 完整命令示例

### 方式 1: 直接运行（推荐用于测试）

```bash
cd /path/to/MM-Reasoning/IMAGEGEN/DataPrep\(CoTT\)/scripts/aaaj_creator

python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --batch-size 100 \
    --summary-output /blob/yasheng/Paper2Fig100k_flux_train_dpg/batch_dpg_summary.json
```

### 方式 2: 使用脚本（推荐用于生产）

```bash
# 修改脚本中的路径
vim run_server_batch.sh

# 运行脚本
bash run_server_batch.sh

# 或者后台运行
nohup bash run_server_batch.sh > batch_processing.log 2>&1 &
```

### 方式 3: 分批处理（处理大量图片时推荐）

```bash
# 第一批：0-10000
python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --max-images 10000 \
    --batch-size 100

# 第二批：10000-20000
python batch_process_directory.py \
    --input-dir /blob/yasheng/Paper2Fig100k_flux_train \
    --output-dir /blob/yasheng/Paper2Fig100k_flux_train_dpg \
    --api-type azure \
    --start-idx 10000 \
    --max-images 10000 \
    --batch-size 100

# 依此类推...
```

## 监控进度

### 查看日志
```bash
tail -f batch_processing.log
```

### 查看 checkpoint
```bash
ls -lh /blob/yasheng/Paper2Fig100k_flux_train_dpg/batch_summary_checkpoint_*.json
```

### 查看已处理的文件数量
```bash
find /blob/yasheng/Paper2Fig100k_flux_train_dpg -name "*_dpg.json" | wc -l
```

### 查看处理统计
```bash
cat /blob/yasheng/Paper2Fig100k_flux_train_dpg/batch_dpg_summary.json | python -m json.tool
```

## 参数说明

- `--input-dir`: 输入目录（包含图片的目录）
- `--output-dir`: 输出目录（DPG JSON 文件保存位置）
- `--api-type`: API 类型（azure 或 openai）
- `--max-images`: 最大处理图片数（用于测试，不设置则处理所有）
- `--start-idx`: 起始索引（用于断点续传）
- `--batch-size`: 每处理多少张保存一次 checkpoint
- `--summary-output`: 最终总结文件路径

## 注意事项

1. **大量图片处理**: 79550 张图片需要很长时间，建议：
   - 使用 screen/tmux 保持会话
   - 定期保存 checkpoint
   - 监控 API 使用量和费用

2. **断点续传**: 脚本会自动跳过已存在的 JSON 文件，可以安全地重新运行

3. **API 限制**: 注意 API 的 rate limit，如果遇到限制，可以：
   - 增加延迟
   - 分批处理
   - 使用多个 API key

4. **磁盘空间**: 确保输出目录有足够的磁盘空间

## 预计时间

假设每张图片处理时间约 3-5 秒：
- 79550 张图片 × 4 秒 = 318200 秒 ≈ 88 小时 ≈ 3.7 天

建议分批处理或使用多进程（如果支持）。

