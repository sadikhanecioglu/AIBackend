# Security Setup Guide 🔒

## Overview

This document provides security best practices for the AI Gateway application. All API keys and sensitive credentials have been removed from the Git repository, and security measures are in place to prevent accidental commits.

## ✅ What's Been Done

### 1. Repository History Cleaned
- **Status**: ✅ Complete
- **Action**: Removed all `.env` files and credentials from Git history using BFG
- **Result**: API keys (OpenAI, Google, Anthropic, etc.) are no longer accessible via Git history
- **Command Used**: `bfg --delete-files ".env*" --delete-folders "credentials"`

### 2. .env Files Configured
- **Status**: ✅ Complete
- **Files**: 
  - `.env.example` - Template with placeholder values (SAFE to commit)
  - `.env` - Actual configuration (NEVER commit, in .gitignore)
  - `.env.*` - Any environment-specific files (NEVER commit, in .gitignore)

### 3. .gitignore Updated
- **Status**: ✅ Complete
- **Rules Added**:
  ```
  # Credentials and Secrets - NEVER COMMIT
  credentials/
  .env
  .env.*
  !.env.example
  .env.local
  .env.*.local
  ```
- **Effect**: Git will refuse to commit any `.env*` files except `.env.example`

### 4. GitHub Repository Rules
- **Status**: ✅ Verified
- **GitHub Push Protection**: Enabled for your repository
- **Result**: GitHub will block any commits containing exposed API keys or secrets

## 📋 Setup Instructions for Developers

### Initial Setup (First Time)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sadikhanecioglu/AIBackend.git
   cd AIBackend
   ```

2. **Create your `.env` file from template**:
   ```bash
   cp .env.example .env
   ```

3. **Add your API keys to `.env`**:
   ```bash
   nano .env
   # or
   vim .env
   ```

4. **Verify Git will not track `.env`**:
   ```bash
   git status
   # .env should NOT appear in the list
   ```

### Updating Configuration

If you need to add new environment variables:

1. **Update `.env.example` FIRST** (without actual values):
   ```bash
   echo "NEW_API_KEY=YOUR_VALUE_HERE" >> .env.example
   ```

2. **Commit `.env.example`**:
   ```bash
   git add .env.example
   git commit -m "Add NEW_API_KEY to configuration template"
   ```

3. **Update your local `.env`** with actual values:
   ```bash
   echo "NEW_API_KEY=your_actual_key_here" >> .env
   ```

## 🔐 Environment Variables Reference

### Required for Operation

```env
# LLM Providers
OPENAI_API_KEY=sk-proj-xxxxx              # OpenAI
ANTHROPIC_API_KEY=sk-ant-xxxxx            # Anthropic  
GOOGLE_API_KEY=xxxxx                      # Google
GOOGLE_APPLICATION_CREDENTIALS=path/to/file  # Google Service Account

# STT/TTS Providers
STT_OPENAI_API_KEY=sk-proj-xxxxx
STT_GOOGLE_API_KEY=xxxxx
STT_ASSEMBLYAI_API_KEY=xxxxx
STT_DEEPGRAM_API_KEY=xxxxx

# Image Generation
REPLICATE_API_TOKEN=xxxxx
TOGETHER_API_KEY=xxxxx
```

### Optional (with defaults)

```env
ENVIRONMENT=development
SERVER_PORT=8000
LOG_LEVEL=INFO
STT_DEFAULT_PROVIDER=openai
```

## ⚠️ Common Security Mistakes to Avoid

### ❌ DON'T

1. **Commit `.env` file**
   ```bash
   # WRONG
   git add .env
   git commit -m "Add environment"
   ```

2. **Hardcode API keys in Python files**
   ```python
   # WRONG
   OPENAI_API_KEY = "sk-proj-xxx"
   
   # RIGHT
   from os import getenv
   OPENAI_API_KEY = getenv("OPENAI_API_KEY")
   ```

3. **Log sensitive data**
   ```python
   # WRONG
   print(f"Using API key: {api_key}")
   logger.info(f"OpenAI key: {openai_api_key}")
   
   # RIGHT
   logger.info("OpenAI initialized")
   ```

4. **Share API keys in chat/emails**
   ```
   # WRONG
   "My OpenAI key is sk-proj-xxxxxxxxxxxxx"
   ```

5. **Store credentials in code comments**
   ```python
   # WRONG
   # API_KEY=sk-proj-xxxxx
   ```

### ✅ DO

1. **Use environment variables**
   ```python
   import os
   api_key = os.getenv("OPENAI_API_KEY")
   ```

2. **Create `.env.example` with placeholders**
   ```env
   OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
   ```

3. **Update `.gitignore` for local files**
   ```
   .env
   .env.*.local
   credentials/
   ```

4. **Rotate keys regularly**
   - Update in `.env` local file
   - No need to commit changes
   - Test to ensure new keys work

5. **Review commits before pushing**
   ```bash
   git diff origin/main
   git status
   ```

## 🔄 If You Accidentally Committed a Secret

### Immediate Actions

1. **Revoke the compromised key immediately**
   - Go to the provider's console (OpenAI, Google, etc.)
   - Regenerate/delete the compromised key
   - Update your `.env` with the new key

2. **Check if it reached GitHub**
   ```bash
   git log --all --full-history -- .env
   ```

3. **If NOT on remote** (only in local commits):
   ```bash
   # Undo the last commit
   git reset --soft HEAD~1
   
   # Or add to .gitignore and commit again
   echo ".env" >> .gitignore
   git add .gitignore
   git commit -m "Add .env to gitignore"
   ```

4. **If ALREADY on remote**:
   - Contact repository maintainer
   - File a security issue
   - May need to force-push cleaned history (use with caution!)

## 📊 Credential Rotation Schedule

| Credential | Rotation Frequency | Last Rotated |
|-----------|-------------------|--------------|
| OpenAI API Key | Quarterly (or when exposed) | [Your Date] |
| Google API Key | Quarterly (or when exposed) | [Your Date] |
| Anthropic API Key | Quarterly (or when exposed) | [Your Date] |
| Database Password | Annually | [Your Date] |
| Service Account Keys | Annually | [Your Date] |

## 🛠️ Tools for Security

### Check if API keys are in Git history
```bash
git log -p --all | grep -i "api_key\|sk-\|secret"
```

### Scan repository for secrets
```bash
# Using truffleHog (requires installation)
truffle-hog filesystem /path/to/repo

# Using git-secrets
git secrets --scan
```

### View what would be committed
```bash
git diff --cached
git diff HEAD
```

## 📞 Need Help?

If you accidentally commit a secret:

1. **Stop immediately** - Revoke the key
2. **Contact maintainer** - Report in private issue
3. **Document the incident** - For future prevention

## 🎓 Further Reading

- [GitHub: Keeping your account and data secure](https://docs.github.com/en/account-and-profile/keeping-your-account-and-data-secure)
- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12-Factor App: Store config in environment](https://12factor.net/config)

---

**Last Updated**: 2025-10-21
**Status**: ✅ All security measures implemented and verified
