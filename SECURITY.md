# Security

- Never commit API keys, tokens, credentials, private datasets, or personal information.
- Keep credentials in environment variables or a managed secret store; do not put them in configs, manifests, logs, or checkpoints.
- Review dataset rows and logs before sharing. The optional PII regex processing is not a complete detector or anonymization guarantee.
- Report suspected vulnerabilities privately to the repository maintainer. Do not include secret values in reports.
