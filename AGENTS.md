# Repository instructions

## Commit authorship

For work performed for the repository owner, attribute new commits to
`Boris Abuzov <ai.sec.boris@gmail.com>` (GitHub: `boris-ai-sec`).
This is the owner's existing Git identity.

Codex and other AI assistants are implementation tools. Do not use an AI tool
as the author, committer, co-author, bot co-author, or attribution trailer unless
the owner explicitly requests that attribution in a future task. Do not add
`Co-authored-by: Codex`, `Co-authored-by: OpenAI`, or equivalent AI trailers.

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
