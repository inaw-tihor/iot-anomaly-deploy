
# DevOps Runbook — Industrial IoT Anomaly Detection

**Version:** 1.1

## Introduction & Scope
This runbook covers deployment, operations, backup & restore, DR replication, security, patching,
and troubleshooting for an Industrial IoT Anomaly Detection system that uses AWS for cloud services
(IoT Core, Greengrass, Kinesis, SageMaker, S3, DynamoDB, SNS, QuickSight) while local CI/CD tools
(Rancher, ArgoCD, Semaphore) run as containers on Docker Desktop for manual controls and approvals.

**Repo:** Operators must clone the repo and run CLI commands from the local copy. Example:
```bash
git clone <repo-url>
cd iot-anomaly-deploy/infra/ansible
ansible-playbook -i inventories/hosts.ini playbooks/01-create-prim-resources.yml
```

## Prerequisites
- AWS CLI v2 installed and configured with a profile (example: iot-admin).
- Docker Desktop running locally.
- Ansible and Terraform installed on the machine or available in the Semaphore runner.
- mosquitto-clients (for MQTT tests), jq, and other utilities.

## User Inputs (placeholders)
Operators must set environment variables or an env file before running playbooks:
```
export AWS_PROFILE="iot-admin"
export AWS_PRIMARY_REGION="us-east-1"
export AWS_DR_REGION="us-west-2"
export AWS_ACCOUNT_ID="<aws-account-id>"
export TENANT_ID="<tenant-id>"
export S3_DATA_LAKE_BUCKET="iot-data-lake-${TENANT_ID}-${AWS_PRIMARY_REGION}"
...
```

## Deployment (CLI-first, manual UI optional)
- The repo contains Ansible playbooks under `infra/ansible/playbooks`.
- CLI commands are provided for each operation; Semaphore jobs are convenience orchestrations.

### Primary bootstrap
From the repo root:
```bash
cd infra/ansible
ansible-playbook -i inventories/hosts.ini playbooks/01-create-prim-resources.yml
ansible-playbook -i inventories/hosts.ini playbooks/02-create-s3-replication-role.yml
ansible-playbook -i inventories/hosts.ini playbooks/03-enable-s3-replication.yml
ansible-playbook -i inventories/hosts.ini playbooks/04-create-dynamodb-global.yml
```

### DR bootstrap
```bash
ansible-playbook -i inventories/hosts.ini playbooks/05-sync-models-to-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/06-create-sagemaker-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/07-create-kinesis-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/08-create-iot-dr.yml
```

### Certificate rollout & device config
```bash
ansible-playbook -i inventories/hosts.ini playbooks/cert_rollout.yml
ansible-playbook -i inventories/hosts.ini playbooks/greengrass_dual_publish.yml
```

## Backup & Restore
Covers S3, DynamoDB, SageMaker artifacts, IoT configs, and Kinesis configuration.

### Backup examples
```bash
aws s3 sync s3://$S3_DATA_LAKE_BUCKET s3://$S3_BACKUP_BUCKET/data-lake --profile $AWS_PROFILE --region $AWS_PRIMARY_REGION
aws dynamodb export-table-to-point-in-time --table-arn arn:aws:dynamodb:${AWS_PRIMARY_REGION}:${AWS_ACCOUNT_ID}:table/${DYNAMODB_TABLE} --s3-bucket $S3_BACKUP_BUCKET --export-format DYNAMODB_JSON --profile $AWS_PROFILE --region $AWS_PRIMARY_REGION
aws sagemaker describe-model --model-name $SAGEMAKER_MODEL_NAME --region $AWS_PRIMARY_REGION --profile $AWS_PROFILE > sagemaker_model_backup.json
aws s3 ls s3://$S3_MODEL_BUCKET --recursive --profile $AWS_PROFILE
```

### Restore examples
```bash
aws s3 sync s3://$S3_BACKUP_BUCKET/data-lake s3://$S3_DATA_LAKE_BUCKET --profile $AWS_PROFILE --region $AWS_PRIMARY_REGION
# DynamoDB import (if using export/import)
aws dynamodb import-table --cli-input-json file://dynamodb_import.json --region $AWS_PRIMARY_REGION --profile $AWS_PROFILE
```

## Security, Patching & Upgrades
- Edge devices: monthly patch windows, Ansible playbooks for `apt-get upgrade`, kernel reboots handled.
- Container images: scan ECR images with Trivy in CI, rebuild + redeploy.
- Local tools: update Rancher/ArgoCD/Semaphore images monthly; snapshot configs before upgrades.
- Notifications: subscribe PagerDuty/Slack to the central SNS topic; automatic Jira ticket creation via Lambda on critical CloudTrail events.

## Monitoring & Observability
- CloudWatch metrics, dashboards for IoT Core, Kinesis, SageMaker.
- Prometheus/Grafana for local containers (Rancher-managed).
- QuickSight for business dashboards.

## Troubleshooting
Commands and playbooks for common issues are included in the repo.

## DR Test & Failover Checklist
A step-by-step checklist to simulate failover, verify data flows in DR, and failback.

(See the `runbook/full_runbook.md` in this repo for the complete detailed runbook text including expected command outputs and SRE checklist.)
