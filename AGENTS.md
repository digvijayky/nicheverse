# AGENTS.md — nicheverse

The full agent/repo policy is in **CLAUDE.md** (same directory). Read it before doing anything.

## The one rule that matters most (git)

NEVER run a git WRITE command in this repository: no `git commit`, `git push`, `git branch`,
`git add`, `git stash`, `git merge`, `git rebase`, `git reset`, `git tag`, or `git commit --amend`.
Sub-agents and automated tools edit the working tree ONLY and then STOP and report their diff.
Only the main orchestrator (top-level session) commits and pushes (it pushes to origin after each
commit); sub-agents and tools never commit or push. Never create branches — everything is on
`main`. Read-only git (`diff`/`log`/`status`) is fine.
