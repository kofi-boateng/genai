#!/bin/bash
set -x #echo on

curl -fsSL https://ollama.com/install.sh | sh

ollama pull mistral
