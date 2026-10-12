# Customer Churn — Kubernetes / Helm / Ingress / Prometheus / Grafana Commands

Project: `D:\\Rahul\\Customer_churn`  
Shell: Windows CMD unless marked PowerShell.  
Cluster: Docker Desktop Kubernetes.  
Last updated: 2026-10-11.

> This is a living runbook. Sections marked **Upcoming** are planned, not completed. We'll update this file as we complete each new step.

---

## 1. Basic cluster checks

```cmd
kubectl get nodes
kubectl get pods -A
kubectl get svc -A
helm list -A
```

## 2. NGINX Ingress

### Add repo and refresh

```cmd
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
```

### Install controller

Installed in `ingress-nginx` with hostPort enabled:

```cmd
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace --set controller.hostPort.enabled=true
```

### Verify and apply app ingress

```cmd
kubectl get pods -n ingress-nginx
kubectl get service -n ingress-nginx
kubectl apply -f k8s\streamlit-ingress.yaml
```

The app was accessed at `http://172.19.0.5` during setup. Local cluster IPs can change.

## 3. Customer Churn Helm chart

Chart: `helm\customer-churn`  
Release: `customer-churn`  
Namespace: `default`

### Lint, install, render

```cmd
helm lint helm\customer-churn
helm install customer-churn helm\customer-churn
helm template customer-churn helm\customer-churn --namespace default > rendered.yaml
```

### Check rendered ServiceMonitor and named Service port

```cmd
findstr /n /c:"kind: ServiceMonitor" rendered.yaml
findstr /n /c:"name: http" rendered.yaml
findstr /n /c:"release: monitoring" rendered.yaml
```

### Upgrade chart

After adding the named `http` port to `helm\customer-churn\templates\api-service.yaml` and creating `helm\customer-churn\templates\api-servicemonitor.yaml`, this was run successfully (release revision 3):

```cmd
helm upgrade customer-churn helm\customer-churn --namespace default --wait --timeout 5m
```

### Verify API resources and image

```cmd
kubectl get servicemonitor -n default
kubectl get svc customer-churn-api -n default -o yaml
kubectl get pods -n default -l app=customer-churn-api
kubectl get deployment customer-churn-api -n default -o jsonpath="{.spec.template.spec.containers[0].image}"
```

Expected Service port is named `http`, with port/targetPort `8000`.

## 4. Prometheus and Grafana installation

### Add repo

```cmd
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm search repo prometheus-community/kube-prometheus-stack
```

### Install kube-prometheus-stack

Chart version `92.3.0`; namespace `monitoring`; values file `monitoring\monitoring-values.yaml`.

```cmd
helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace --values monitoring\monitoring-values.yaml --version 92.3.0 --wait --timeout 10m
```

An upgrade using the same chart version and values was also run:

```cmd
helm upgrade monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --values monitoring\monitoring-values.yaml --version 92.3.0 --wait --timeout 10m
```

### Verify monitoring components

```cmd
kubectl get pods -n monitoring
kubectl get pvc -n monitoring
helm list -n monitoring
```

Keep the Grafana PVC `monitoring-grafana`; deleting it can remove persisted Grafana data.

## 5. Access Grafana and Prometheus

Run each port-forward in a separate CMD window and leave it running.

### Grafana

```cmd
kubectl port-forward -n monitoring service/monitoring-grafana 3000:80
```

URL: `http://localhost:3000`  
Username: `admin`

Get the encoded password:

```cmd
kubectl get secret monitoring-grafana -n monitoring -o jsonpath="{.data.admin-password}"
```

Decode in PowerShell (replace `BASE64_OUTPUT` with the result):

```powershell
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String("BASE64_OUTPUT"))
```

Do not commit credentials. Change the password if it has been exposed.

### Prometheus

```cmd
kubectl port-forward -n monitoring service/monitoring-kube-prometheus-prometheus 9090:9090
```

URLs:
- Prometheus: `http://localhost:9090`
- Targets: `http://localhost:9090/targets`
- Query UI: `http://localhost:9090/graph`

## 6. Prometheus selectors and PromQL

### Check selectors

```cmd
kubectl get prometheus monitoring-kube-prometheus-prometheus -n monitoring -o jsonpath="{.spec.serviceMonitorSelector.matchLabels}"
kubectl get prometheus monitoring-kube-prometheus-prometheus -n monitoring -o jsonpath="{.spec.serviceMonitorNamespaceSelector}"
```

Observed selector: `{"release":"monitoring"}`. The namespace selector output was `{}`, so ServiceMonitors across namespaces are selected.

### Queries used/tested

Run these in `http://localhost:9090/graph` and click Execute.

```promql
up
```

```promql
sum by (job) (up)
```

```promql
up == 0
```

```promql
http_requests_total
```

```promql
sum by (handler) (rate(http_requests_total{job="customer-churn-api"}[5m]))
```

```promql
sum by (status) (rate(http_requests_total{job="customer-churn-api"}[5m]))
```

```promql
up{job="customer-churn-api"}
```

P95 latency query to use once confirmed working:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket{job="customer-churn-api"}[5m]))
)
```

If a query returns no data, check that the metric exists and that label selectors match actual series.

## 7. FastAPI metrics instrumentation

Files changed:
- `src\customer_churn\app.py`
- `requirements-api.txt`

Dependency added to `requirements-api.txt`:

```text
prometheus-fastapi-instrumentator
```

Instrumentation in `src\customer_churn\app.py`, after `app = FastAPI(...)`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app=app).expose(app=app, endpoint="/metrics")
```

### Test local API metrics (local port 8001)

```cmd
curl -i http://127.0.0.1:8001/metrics
```

### Test in a Kubernetes API pod

```cmd
kubectl exec -n default deployment/customer-churn-api -- python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/metrics').status)"
```

Expected status: `200`. This returned `200` after the new image was deployed.

### Confirm target health

Open `http://localhost:9090/targets`. The `customer-churn-api` ServiceMonitor target was confirmed **2/2 UP**.

## 8. Upcoming — update this section as we finish each task

These are planned, not yet completed:

- [ ] Check Grafana **Connections → Data sources** for an existing Prometheus data source.
- [ ] If needed, configure Prometheus as a Grafana data source. The in-cluster URL to try is `http://monitoring-kube-prometheus-prometheus.monitoring.svc:9090`; test the connection.
- [ ] Create and save a Customer Churn API dashboard with request rate, HTTP response rates by status, availability, and p95 latency.
- [ ] Record any dashboard import/export or backup commands actually used.
- [ ] Add alert rules after dashboard queries have been verified.
- [ ] Add commands for any further monitoring components only after they are actually configured.

## 9. Maintenance notes

- Use Helm for Helm-managed resources; avoid manual patches unless necessary.
- Keep secrets, tokens, and decoded passwords out of Git.
- Run `git status --short` before staging changes; generated files such as `rendered.yaml` may be untracked.
- Port-forward sessions end when their terminal closes.
- Keep a clear distinction between commands already run and commands that are only planned.
