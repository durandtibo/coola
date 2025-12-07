# Security Policy

## Supported Versions

We release patches for security vulnerabilities. The following versions are currently supported:

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| 0.8.x   | :white_check_mark: |
| < 0.8.0 | :x:                |

## Reporting a Vulnerability

The `coola` team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**durand.tibo+gh@gmail.com**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

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

### What to Expect

- We will acknowledge receipt of your vulnerability report within 2 business days
- We will send a more detailed response within 7 days indicating the next steps
- We will work with you to understand the scope and severity of the issue
- We will keep you informed of the progress towards a fix and full announcement
- We may ask for additional information or guidance

### Disclosure Policy

- Security issues will be kept confidential until a fix is released
- We will coordinate the disclosure with you
- Once a fix is released, we will credit you for the discovery (unless you prefer to remain anonymous)

### Security Update Policy

- Security updates will be released as soon as possible
- Security updates will be clearly marked in release notes
- We will notify users through GitHub releases and other appropriate channels

## Security Best Practices

When using `coola` in your projects:

1. **Keep dependencies updated**: Regularly update to the latest version of `coola` and its dependencies
2. **Monitor security advisories**: Watch the repository for security announcements
3. **Review code**: When comparing sensitive data, ensure appropriate access controls are in place
4. **Validate inputs**: Always validate data before comparison, especially when handling user inputs

## Known Security Considerations

`coola` is a library for comparing complex objects. When using it:

- **Resource exhaustion**: Be cautious when comparing very large or deeply nested data structures, as this may consume significant memory and CPU resources
- **Sensitive data**: When comparing objects containing sensitive data, ensure proper access controls and logging are in place
- **Third-party data structures**: When extending `coola` to support custom types, ensure proper validation and error handling

## Bug Bounty Program

At this time, we do not offer a paid bug bounty program. However, we deeply appreciate security researchers who help keep `coola` and our users safe.

## Attribution

We would like to thank the following security researchers for responsibly disclosing vulnerabilities:

- (List will be updated as security issues are reported and resolved)

## Contact

For any questions about this security policy, please contact: durand.tibo+gh@gmail.com
