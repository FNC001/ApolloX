#!/bin/bash
# gen_env.sh - gnenerate .env

CUR_DIR=$(pwd)

cat > .env << EOF
PROJECT_ROOT="${CUR_DIR}"
HYDRA_JOBS="${CUR_DIR}/log"
EOF

echo ".env has been generated"
cat .env

