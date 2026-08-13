# Security policy

This repository is an experimental game and machine-learning research project. It does not currently run a network service or process untrusted model files by default.

Please report a suspected vulnerability privately through GitHub's **Report a vulnerability** feature rather than opening a public issue. Include the affected component, reproduction steps, and potential impact. There is currently no guaranteed response-time service level.

Never commit Play signing keys, credentials, tokens, personal data, or untrusted serialized model files. PyTorch checkpoints can execute code during unsafe deserialization; only load checkpoints from sources you trust.
