#!/bin/bash
# gen_env.sh - 自动生成 .env 文件，写入当前目录相关路径

CUR_DIR=$(pwd)

cat > .env << EOF
PROJECT_ROOT="${CUR_DIR}"
HYDRA_JOBS="${CUR_DIR}/log"
EOF

echo ".env has been generated"
cat .env

