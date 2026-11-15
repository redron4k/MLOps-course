#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.tools/kind load docker-image hw11-api:v2 --name hw11
kubectl --kubeconfig .tools/kubeconfig create namespace hw11 --dry-run=client -o yaml | kubectl --kubeconfig .tools/kubeconfig apply -f -
kubectl --kubeconfig .tools/kubeconfig apply -f monitoring/metrics-server.yaml
kubectl --kubeconfig .tools/kubeconfig -n kube-system patch deployment metrics-server --type=json -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--metric-resolution=15s"}]'
kubectl --kubeconfig .tools/kubeconfig apply -f monitoring/stack.yaml
kubectl --kubeconfig .tools/kubeconfig -n hw11 rollout status deployment/prometheus --timeout=180s
kubectl --kubeconfig .tools/kubeconfig -n hw11 rollout status deployment/grafana --timeout=180s
kubectl --kubeconfig .tools/kubeconfig -n kube-system rollout status deployment/metrics-server --timeout=180s
