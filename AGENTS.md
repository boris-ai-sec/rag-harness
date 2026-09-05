# Repository instructions

## Commit authorship

For work performed for the repository owner, attribute new commits to
`Boris Abuzov <ai.sec.boris@gmail.com>` (GitHub: `boris-ai-sec`).
This is the owner's existing Git identity.

Automated implementation tools are tools, not repository contributors by
default. Unless the owner explicitly requests otherwise, do not use an automated
implementation tool as the author, committer, co-author, bot co-author, or
attribution trailer. Do not add automated-tool attribution trailers.

Before committing, check both `git var GIT_AUTHOR_IDENT` and
`git var GIT_COMMITTER_IDENT`; environment overrides can supersede Git config.
After committing, inspect `git show -s --format=fuller HEAD` and all message
trailers. Inspect the final squash message and resulting author when merging.
Follow the workflow in `CONTRIBUTING.md`.

## Historical integrity

Do not rewrite existing commits to normalize authorship. Preserve published
tags, releases, frozen snapshots, evidence, provenance records, and checksum
manifests. Authorship policy changes do not change Framework or laboratory
conclusions, evidence authority, or licensing scope.
