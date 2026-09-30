# Security Policy
This document outlines the security policy, vulnerability disclosure guidelines, supported versions, and reporting procedures for the StellarAI open-source project. All security-related practices apply to both the GitHub source repository and the HuggingFace model release repository.
Supported Versions
The following project versions are currently maintained and receive security patches, bug fixes, and stability updates. Legacy or unofficial builds are not supported.
Version
Supported
Beta-0.05b (Latest Stable)
Active Support
Pre-release / Dev Builds
Partial Support
Outdated Beta Versions
No Support
Security Features
StellarAI adopts multiple security mechanisms to ensure safe deployment and usage in local and offline environments:
- Safetensors Default Weight Format: Official model weights are released in Safetensors format to prevent arbitrary code execution, file injection, and pickle-based security risks.
- Offline-First Design: The model supports fully local inference and training without mandatory network data collection or remote server communication.
- Open Source Auditability: All core model code, tokenizer logic, and training scripts are fully open for public security review and inspection.
- Local Resource Isolation: The project runs entirely in user-local environments with no background data transmission behavior.
Known Limitations & Security Notes
- Legacy PyTorch .pt checkpoint files are provided only for training snapshot compatibility. Users are strongly recommended to use Safetensors for formal deployment to avoid deserialization risks.
- This model is intended for academic research, local testing, and non-commercial experimental use. It is not designed for high-risk production scenarios involving sensitive user data.
- Improper modification of model architecture, tokenizer rules, or external dataset injection may introduce abnormal output risks. Users shall ensure the legality and safety of input content and custom training data.
- No built-in content filtering module is enabled by default. Deployers are responsible for adding necessary content moderation mechanisms for public-facing services.
Vulnerability Reporting
If you discover any security vulnerability, loophole, abnormal code behavior, or potential security risk within the StellarAI project, please submit a responsible disclosure through the official GitHub Issue channel with the security label.
Please include the following information in your report:
- A detailed description of the vulnerability and reproduction steps
- Environment information (system version, dependency version, project build version)
- Potential impact scope and risk level
- Suggested fixes or mitigation methods (if available)
All valid security reports will be reviewed, tracked, and fixed in subsequent project updates. We strictly follow responsible disclosure principles and will not publicly disclose unpatched vulnerabilities.
Security Update Policy
- Critical security vulnerabilities will be prioritized fixed and patched in the latest supported version.
- Minor security optimization updates will be included in routine version iterations.
- All security-related fixes will be clearly recorded in the project release notes.
Disclaimer
StellarAI is an open-source experimental project. The development team is not liable for any direct or indirect losses caused by unauthorized commercial deployment, improper use, or secondary modification of this project. Users must comply with local laws, regulations, and ethical guidelines during all usage processes.
