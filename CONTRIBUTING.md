# Owner contribution workflow

## Configure the local identity

For owner-directed work, configure this repository locally:

```sh
git config --local user.name "Boris Abuzov"
git config --local user.email "ai.sec.boris@gmail.com"
git config --local user.useConfigOnly true
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

Both identities must resolve to Boris Abuzov and the email above, associated
with `boris-ai-sec`. Resolve conflicting `GIT_AUTHOR_*` or `GIT_COMMITTER_*`
environment overrides before creating a commit. Do not change another human
contributor's attribution to the owner.

Automated implementation tools receive no author, committer, co-author, bot
co-author, or attribution trailer by default. An explicit future owner request
is required for an exception.

## Review and merge

1. Create a normal topic branch from current `main` and make a bounded change.
2. Run `pytest -q` with the project dependencies installed. For the frozen
   Framework conformance check, set `FRAMEWORK_V03_ROOT` to the extracted
   Technical Prototype V0.3. Report unavailable service checks and skips.
3. Review the diff and inspect `git show -s --format=fuller HEAD`. Inspect
   trailers with `git log -1 --format=%B | git interpret-trailers --parse`.
   The owner-work commit author must be Boris Abuzov, with no automated-tool
   attribution.
4. Push the topic branch and open a pull request. Validate the exact PR head.
5. Squash merge using the owner's GitHub account. Inspect the proposed squash
   message explicitly: generated messages can carry trailers from branch
   commits. Ensure the squash author is Boris Abuzov and no automated-tool
   attribution is present in the message. GitHub may supply its normal web-flow
   committer.
6. Verify the resulting commit's author, message, parent, validation result,
   and public attribution, then update the local checkout and check it is clean.

## Preserve evidence

Never amend published history, force push, move published tags, or regenerate
historical evidence to change attribution. Keep frozen source snapshots,
SHA256SUMS, provenance, and existing evidence byte-identical. A new workflow
commit does not supersede a frozen baseline or alter a Framework/Lab conclusion
or the repository's licensing scope.
