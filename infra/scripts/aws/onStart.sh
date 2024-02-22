#!/bin/bash
set -x #echo on

set -e

#sudo -u ec2-user -i <<'EOF'

curl -fsSL https://ollama.com/install.sh | sh

ollama pull mistral

EOF
