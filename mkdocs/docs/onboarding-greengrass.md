---
id: onboard-gg
title: "Tenant Onboarding — Track B: Greengrass"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [greengrass, nucleus, role alias, components]
---

# Tenant Onboarding — Track B: Greengrass (edge features)

## Install Nucleus (auto-provision path)
```bash
sudo apt update && sudo apt install -y default-jdk
cd ~ && curl -s -o gg.zip https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip
unzip gg.zip -d GreengrassInstaller && rm gg.zip

sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE   -jar ./GreengrassInstaller/lib/Greengrass.jar   --aws-region "$AWS_PRIMARY_REGION"   --thing-name "${TENANT_ID}-gg-core"   --tes-role-alias-name "GreengrassCoreTokenExchangeRoleAlias"   --component-default-user ggc_user:ggc_group   --provision true --setup-system-service true --deploy-dev-tools true
```

## Deploy components
- `aws.greengrass.Cli`
- `aws.greengrass.TokenExchangeService`
- `aws.greengrass.StreamManager`
- (Optional) your custom edge component.
