# AGENTS.md — nicheverse

The full agent/repo policy is in **CLAUDE.md** (same directory). Read it before doing anything.

## The one rule that matters most (git)

NEVER run a git WRITE command in this repository: no `git commit`, `git push`, `git branch`,
`git add`, `git stash`, `git merge`, `git rebase`, `git reset`, `git tag`, or `git commit --amend`.
Sub-agents and automated tools edit the working tree ONLY and then STOP and report their diff.
Only the human's main orchestrator commits, and only the human pushes, and only when explicitly
asked. Never create branches — everything is on `main`. Read-only git (`diff`/`log`/`status`) is fine.
