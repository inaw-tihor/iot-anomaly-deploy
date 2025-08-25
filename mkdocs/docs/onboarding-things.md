---
id: onboard-things
title: "Tenant Onboarding — Track A: IoT Things"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [onboarding, iot core, certs, policy, rule]
---

# Tenant Onboarding — Track A: IoT Things (no Greengrass)

## Provision identity
```bash
THING_NAME="${TENANT_ID}-device1"

aws iot create-thing --thing-name "$THING_NAME"   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE"

aws iot create-keys-and-certificate --set-as-active   --certificate-pem-outfile cert-${THING_NAME}.pem   --public-key-outfile public-${THING_NAME}.key   --private-key-outfile private-${THING_NAME}.key   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE" | tee cert-output.json
export CERT_ARN=$(jq -r '.certificateArn' cert-output.json)
```

## Attach least-privilege IoT policy
Use your repo policy template and attach to the certificate, then attach the cert to the Thing.

## Route telemetry
Create/verify an IoT Rule to fan out to **Kinesis** (+ optional **Lambda**). Use topic pattern:  
`tenants/${TENANT_ID}/devices/+/telemetry`

## Device test publish
```bash
IOT_ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:Data-ATS   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE" --query endpointAddress -o text)

mosquitto_pub -h "$IOT_ENDPOINT" -p 8883   --cafile AmazonRootCA1.pem   --cert cert-${THING_NAME}.pem   --key private-${THING_NAME}.key   -t "tenants/${TENANT_ID}/devices/${THING_NAME}/telemetry"   -m '{"ts":'$(date +%s)',"temperature":83.2,"vibration":0.19}' -q 1
```
