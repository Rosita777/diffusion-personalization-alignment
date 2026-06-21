# GitHub Backup Setup

Date: 2026-06-21

Remote repository:

```text
https://github.com/Rosita777/diffusion-personalization-alignment
```

Current status:

- The local project was initialized as a git repository on branch `main`.
- `token.txt` exists locally for GitHub authentication but is ignored by git and must not be committed.
- The remote repository is public, so secrets, raw dataset caches, model checkpoints, and large generated artifacts should stay out of git.
- Direct git HTTPS access to `github.com:443` timed out from this machine during setup.
- `api.github.com` was reachable, so the initial remote backup was created through the GitHub Git Data API.

Remote commits created during setup:

- `f45a9a941e60465b1517ded5da757f323712d6ff`: initialize `README.md`.
- `c2197f90b5e36969d50f6dbc3fcaa8c02ebd31df`: add project structure and hygiene rules.

Important caution:

Because the initial remote backup used the GitHub API instead of `git push`, the local git commit hash and remote commit hash may differ. Before relying on ordinary `git push`, first check whether direct GitHub access works and reconcile local history with `origin/main` safely. Do not force-push without confirming that no remote-only work will be lost.
