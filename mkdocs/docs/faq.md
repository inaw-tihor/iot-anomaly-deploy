---
id: faq
title: "Operator FAQs"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [where playbooks, validate, aws ra]
---

# Operator FAQs

**Where are the playbooks?**  
`infra/ansible/playbooks/*.yml` in the infra repo you cloned.

**How do I validate replication quickly?**  
Run `validate/validate_replication.sh` in the repo and the Kinesis `get-records` one-liner.

**Which AWS reference is this aligned to?**  
Industrial Anomaly Detection RA (hot/warm/cold).
