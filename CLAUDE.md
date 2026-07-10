# nicheverse — repository policy for Claude / agents

## GIT POLICY — READ FIRST. Applies to EVERY agent, sub-agent, and automated tool working in this repo.

- **NEVER run any git WRITE command in this repository.** Do NOT `git commit`, `git push`,
  `git branch`, `git checkout -b`, `git add`, `git stash`, `git merge`, `git rebase`,
  `git reset`, `git tag`, `git commit --amend`, or any other command that changes the repo
  state, refs, or history. This is absolute and applies to all background/sub agents and tools.
- **Make edits to the working tree ONLY, then STOP and report your diff.** Exactly ONE actor
  commits: the human-driven main orchestrator (the top-level Claude session), and only on the
  human's behalf. If you believe a commit is warranted, SAY SO in your report — do not do it.
- **Only the main orchestrator (top-level Claude session) commits AND pushes; it pushes after
  every commit** (the human wants `main` mirrored to GitHub continuously). A sub-agent, background
  agent, or automated tool must NEVER commit or push under any circumstance - it edits and reports.
- **NEVER create branches.** All work is on `main`; there must be nothing beyond `main` and
  `origin/main` (`git branch -a` to verify). If a branch ever exists, tell the human.
- Read-only git is fine: `git diff`, `git log`, `git status`, `git show`.
- **Why:** an agent that commits or pushes on its own corrupts the shared history and can put
  unreviewed, badly-messaged commits onto GitHub. This happened once (a stray "refactoring"
  commit was auto-committed and pushed). Do not repeat it.

## Package facts (context)

- Editable install (`pip install -e`): edits to `src/nicheverse/` are live with no reinstall.
- Run tests via SLURM (`sbatch --partition=preemptable --requeue`), NEVER on the login node.
- Everything lives on `main`; the human commits in small, well-messaged, logical units.
