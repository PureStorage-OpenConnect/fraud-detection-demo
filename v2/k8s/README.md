# Kubernetes Deployment for Fraud Detection Demo

## Prerequisites

1. **Kubernetes cluster** with:
   - At least one GPU node with NVIDIA GPU Operator or nvidia-device-plugin
   - At least one CPU node
   - StorageClass for PVCs (FlashBlade recommended)

2. **NVIDIA GPU Operator** installed:
   ```bash
   # Check GPU availability
   kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu"
   ```

3. **Container images** available (either from GitHub registry or local):
   - `ghcr.io/purestorage-openconnect/fraud-detection-demo/dashboard:latest`
   - `ghcr.io/purestorage-openconnect/fraud-detection-demo/worker-cpu:latest`
   - `ghcr.io/purestorage-openconnect/fraud-detection-demo/worker-gpu:latest`

## Quick Start

### 1. Label your nodes

```bash
# Label CPU node
kubectl label node slc6-lg-n3-b30-25 node-type=cpu

# Label GPU node(s)
kubectl label node <gpu-node-name> node-type=gpu
```

### 2. Update StorageClass (optional)

Edit `pvc.yaml` and uncomment/set your StorageClass:
```yaml
storageClassName: pure-file  # For FlashBlade
# or
storageClassName: pure-block  # For FlashArray
```

### 3. Deploy

```bash
# Using kustomize
kubectl apply -k .

# Or apply individual files
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f worker-cpu-deployment.yaml
kubectl apply -f worker-gpu-deployment.yaml
kubectl apply -f dashboard-deployment.yaml
```

### 4. Copy data to PVCs

You need to copy the transaction data to the PVCs:

```bash
# Get the PVC pod names or create a temporary pod
kubectl run data-loader --image=busybox -n fraud-demo --rm -it --restart=Never -- sh

# Or use kubectl cp with a running pod
kubectl cp ./data/cpu/transactions.parquet fraud-demo/worker-cpu-<pod-id>:/data/
kubectl cp ./data/gpu/transactions.parquet fraud-demo/worker-gpu-<pod-id>:/data/
```

### 5. Access the Dashboard

```bash
# Get NodePort URL
kubectl get svc dashboard -n fraud-demo

# Access at http://<any-node-ip>:30080
```

## Verify Deployment

```bash
# Check all pods
kubectl get pods -n fraud-demo

# Check services
kubectl get svc -n fraud-demo

# View logs
kubectl logs -f deployment/dashboard -n fraud-demo
kubectl logs -f deployment/worker-cpu -n fraud-demo
kubectl logs -f deployment/worker-gpu -n fraud-demo
```

## GPU Verification

```bash
# Check if GPU is allocated
kubectl describe pod -l component=worker-gpu -n fraud-demo | grep -A5 "Limits:"

# Check GPU node resources
kubectl describe node <gpu-node> | grep -A10 "Allocated resources:"
```

## Troubleshooting

### No GPU available
```bash
# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Check node GPU capacity
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.capacity["nvidia.com/gpu"]}'
```

### Pod not scheduling
```bash
# Check events
kubectl describe pod <pod-name> -n fraud-demo

# Common issues:
# - Node selector not matching
# - Insufficient GPU resources
# - PVC not bound
```

### Worker not connecting
```bash
# Check service discovery
kubectl exec -it deployment/dashboard -n fraud-demo -- curl http://worker-cpu:5001/health
kubectl exec -it deployment/dashboard -n fraud-demo -- curl http://worker-gpu:5002/health
```

## Cleanup

```bash
kubectl delete namespace fraud-demo
```
