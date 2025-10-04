#!/usr/bin/env bash

# 测试SSH连接和文件访问
# 验证SSH密钥和路径配置是否正确

set -euo pipefail

# 配置
LOCAL_HOST="suny0a@10.64.74.69"
LOCAL_SSH_KEY="~/.ssh/id_rsa"
LOCAL_PDF_DIR="/home/suny0a/arxiv_dataset/pdf/"
LOCAL_OUT_DIR="/home/suny0a/arxiv_dataset/md/"

echo "=== 测试SSH连接和文件访问 ==="
echo "本地存储机器: ${LOCAL_HOST}"
echo "SSH密钥: ${LOCAL_SSH_KEY}"
echo "PDF目录: ${LOCAL_PDF_DIR}"
echo "输出目录: ${LOCAL_OUT_DIR}"
echo ""

# 1. 测试SSH连接
echo "1. 测试SSH连接..."
if ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "echo 'SSH连接成功'"; then
    echo "✓ SSH连接正常"
else
    echo "✗ SSH连接失败，请检查SSH密钥和网络连接"
    exit 1
fi

# 2. 测试PDF目录访问
echo "2. 测试PDF目录访问..."
if ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "test -d '${LOCAL_PDF_DIR}' && echo 'PDF目录存在' || echo 'PDF目录不存在'"; then
    echo "✓ PDF目录可访问"
else
    echo "✗ 无法访问PDF目录"
    exit 1
fi

# 3. 统计PDF文件数量
echo "3. 统计PDF文件数量..."
PDF_COUNT=$(ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "find '${LOCAL_PDF_DIR}' -name '*.pdf' -type f | wc -l")
echo "找到 ${PDF_COUNT} 个PDF文件"

# 4. 显示前几个PDF文件
echo "4. PDF文件示例:"
ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "find '${LOCAL_PDF_DIR}' -name '*.pdf' -type f | head -3"

# 5. 测试输出目录创建
echo "5. 测试输出目录创建..."
if ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "mkdir -p '${LOCAL_OUT_DIR}' && echo '输出目录已准备'"; then
    echo "✓ 输出目录已准备"
else
    echo "✗ 无法创建输出目录"
    exit 1
fi

# 6. 测试文件下载速度
echo "6. 测试文件下载速度..."
SAMPLE_PDF=$(ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "find '${LOCAL_PDF_DIR}' -name '*.pdf' -type f | head -1")
if [[ -n "$SAMPLE_PDF" ]]; then
    echo "测试下载: $SAMPLE_PDF"
    time scp -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}:${SAMPLE_PDF}" "/ibex/user/suny0a/arxiv_dataset/test_download.pdf" 2>/dev/null
    if [[ -f "/ibex/user/suny0a/arxiv_dataset/test_download.pdf" ]]; then
        FILE_SIZE=$(stat -c%s "/ibex/user/suny0a/arxiv_dataset/test_download.pdf" 2>/dev/null || stat -f%z "/ibex/user/suny0a/arxiv_dataset/test_download.pdf" 2>/dev/null)
        echo "✓ 下载成功，文件大小: ${FILE_SIZE} 字节"
        rm -f "/ibex/user/suny0a/arxiv_dataset/test_download.pdf"
    else
        echo "✗ 下载失败"
    fi
fi

echo ""
echo "=== 测试完成 ==="
echo "如果所有测试都通过，可以运行主脚本:"
echo "bash srun_infer_correct.sh"
