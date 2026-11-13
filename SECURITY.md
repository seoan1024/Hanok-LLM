# Security

## Protect local data and credentials

- Never commit API keys, access tokens, credentials, private datasets, or personal information.
- Keep secrets in environment variables or a managed secret store. Do not put them in source files, configuration, dataset manifests, logs, or checkpoints.
- Review downloaded dataset rows, prompts, logs, and evaluation reports before sharing. They may contain sensitive content.
- The training code does not provide a PII detector or anonymization guarantee. Dataset users are responsible for privacy review and permitted use.

## Report a vulnerability

Do not publish exploitable details in a public issue. Contact the repository maintainer privately through a trusted channel associated with the repository and include a concise description, affected version or commit, impact, and reproduction steps. Do not include secret values or personal data in the report. This repository does not currently publish a dedicated security contact or response-time commitment.
