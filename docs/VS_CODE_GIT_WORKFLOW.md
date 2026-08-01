# VS Code and Git workflow

This repository is a working copy of `DeepWave-KAUST/PINNslope` prepared for the dual-angle PINN adaptation.

## Local open command

From PowerShell:

```powershell
code "C:\Users\HW\Documents\Codex\2026-08-01\clone-https-github-com-deepwave-kaust\PINNslope"
```

Or in VS Code, use `File > Open Folder...` and select the `PINNslope` folder.

## Remote layout

- `upstream`: original paper repository, read-only for our work.
- `origin`: your new GitHub repository, to be added after it exists.
- `feature/dual-angle`: working branch for the four-network + loss changes.

To add your own remote repository later:

```powershell
git remote add origin https://github.com/<your-user>/PINNslope-dual-angle.git
git push -u origin main
git push -u origin feature/dual-angle
```

Create the GitHub repository as an empty repository first. Do not add a README, license, or `.gitignore` on GitHub, because this local repository already has them.

## Daily collaboration loop

```powershell
git status --short --branch
git switch feature/dual-angle
git pull --ff-only origin feature/dual-angle
```

After changes:

```powershell
git diff
git add <files>
git commit -m "Describe the change"
git push
```

Use VS Code's Source Control panel to review diffs before committing.
