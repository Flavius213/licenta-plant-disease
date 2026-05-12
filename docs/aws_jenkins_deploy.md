# Deploy AWS cu GitHub si Jenkins

## Arhitectura

Fluxul recomandat pentru proiect:

1. Codul este pus pe GitHub.
2. GitHub trimite webhook catre Jenkins la fiecare push.
3. Jenkins face checkout la cod.
4. Jenkins ruleaza un static check Python.
5. Jenkins construieste imaginea Docker `plant-diagnosis-api`.
6. Jenkins copiaza imaginea pe o instanta AWS EC2 prin SSH.
7. EC2 ruleaza containerul Docker si expune API-ul pe portul `8000`.

## Serviciul API

Aplicatia cloud este in:

```text
app/api.py
```

Endpoint-uri:

```text
GET  /health
GET  /classes
POST /diagnose
```

Exemplu local:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Test:

```powershell
curl http://localhost:8000/health
```

## Docker local

Build:

```powershell
docker build -t plant-diagnosis-api:local .
```

Run:

```powershell
docker run --rm -p 8000:8000 plant-diagnosis-api:local
```

## AWS EC2

Configurare minima:

- Ubuntu Server LTS
- Security Group cu porturile:
  - `22` pentru SSH
  - `8000` pentru API
- Cheie SSH `.pem`

Nu pune cheia SSH in GitHub.

## Jenkins

Plugin-uri utile:

- Git
- Pipeline
- SSH Agent

Credential necesar in Jenkins:

```text
ID: aws-ec2-ssh-key
Type: SSH Username with private key
Username: ubuntu
Private key: cheia .pem pentru EC2
```

Pipeline-ul este in:

```text
Jenkinsfile
```

Parametri Jenkins:

```text
EC2_HOST = DNS/IP instanta EC2
EC2_USER = ubuntu
SSH_CREDENTIALS_ID = aws-ec2-ssh-key
APP_PORT = 8000
```

## GitHub webhook

In GitHub repository:

```text
Settings -> Webhooks -> Add webhook
Payload URL: http://IP_JENKINS:8080/github-webhook/
Content type: application/json
Events: Just the push event
```

## Model

Dockerfile include doar:

```text
models/classifier/best_model.pt
```

Datasetul `data/` nu este pus in GitHub si nu este inclus in Docker image.

## Observatie importanta

Prima rulare cu `remove_background=true` poate fi mai lenta deoarece `rembg` incarca modelul de segmentare. Pentru productie simpla poti trimite:

```text
POST /diagnose?remove_background=false
```

Pentru pipeline-ul complet al lucrarii foloseste:

```text
POST /diagnose?remove_background=true&multi_crop=true&max_crops=8
```
