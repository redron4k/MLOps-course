#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .tools
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=darwin-arm64 ;;
  Linux-x86_64) platform=linux-amd64 ;;
  Linux-aarch64) platform=linux-arm64 ;;
  *) echo 'Install kind 0.27.0 and Helm 3.17.3 for your platform'; exit 1 ;;
esac
curl -fL "https://kind.sigs.k8s.io/dl/v0.27.0/kind-$platform" -o .tools/kind
curl -fL "https://kind.sigs.k8s.io/dl/v0.27.0/kind-$platform.sha256sum" -o .tools/kind.sha256
expected=$(cut -d ' ' -f 1 .tools/kind.sha256)
actual=$(shasum -a 256 .tools/kind | cut -d ' ' -f 1)
[ "$actual" = "$expected" ]
chmod +x .tools/kind
curl -fL "https://get.helm.sh/helm-v3.17.3-$platform.tar.gz" -o .tools/helm.tar.gz
curl -fL "https://get.helm.sh/helm-v3.17.3-$platform.tar.gz.sha256sum" -o .tools/helm.sha256
expected=$(cut -d ' ' -f 1 .tools/helm.sha256)
actual=$(shasum -a 256 .tools/helm.tar.gz | cut -d ' ' -f 1)
[ "$actual" = "$expected" ]
tar -xzf .tools/helm.tar.gz -C .tools
ln -sf "$platform/helm" .tools/helm
