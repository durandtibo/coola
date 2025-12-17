# Security Policy

## Supported Versions

We release patches for security vulnerabilities. The following versions are currently supported:

| Version | Supported          |
|---------|--------------------|
| 0.9.x   | :white_check_mark: |
| < 0.9.0 | :x:                |

## Reporting a Vulnerability

The `coola` team takes security bugs seriously. We appreciate your efforts to responsibly disclose
your findings.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**durand.tibo+gh@gmail.com**

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

### Security Update Policy

- Security updates will be released as soon as possible
- Security updates will be clearly marked in release notes
- We will notify users through GitHub releases and other appropriate channels

## Security Best Practices

When using `coola` in your projects:

1. **Keep dependencies updated**: Regularly update to the latest version of `coola` and its
   dependencies
2. **Monitor security advisories**: Watch the repository for security announcements
3. **Review code**: When comparing sensitive data, ensure appropriate access controls are in place
4. **Validate inputs**: Always validate data before comparison, especially when handling user inputs

## Known Security Considerations

`coola` is a library for comparing complex objects. When using it:

- **Resource exhaustion**: Be cautious when comparing very large or deeply nested data structures,
  as this may consume significant memory and CPU resources
- **Sensitive data**: When comparing objects containing sensitive data, ensure proper access
  controls and logging are in place
- **Third-party data structures**: When extending `coola` to support custom types, ensure proper
  validation and error handling
