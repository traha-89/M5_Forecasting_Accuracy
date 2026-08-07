---
name: pr-summary
description: Reviews all open pull requests on this repo — fetches each PR's diff, comments, reviews, and changed files via the GitHub CLI, then produces a concise summary and assessment. Use when the user asks to "review open PRs", "summarize the PRs", "what's open right now", or similar.
disable-model-invocation: true
allowed-tools: Bash(gh *)
context: fork
---

# PR Summary

Review every open pull request on this repository and report a concise summary of each.

## Prerequisites

Check the GitHub CLI is available and authenticated before doing anything else:

```
gh auth status
```

If `gh` is missing or not authenticated, stop and tell the user how to fix it
(`winget install GitHub.cli` on Windows, then `gh auth login`) rather than guessing.

## Steps

1. **List open PRs**

   ```
   !`gh pr list --state open --json number,title,author,headRefName,baseRefName,url,createdAt,updatedAt,isDraft`
   ```

   If there are no open PRs, say so and stop.

2. **For each open PR, gather full context:**

   - Description/metadata: !`gh pr view <number> --json body,additions,deletions,changedFiles,files,comments,reviews,statusCheckRollup`
   - Full diff: !`gh pr diff <number>`

   Fetch every PR before writing any summaries, so the report covers all of them consistently.

3. **Review each PR's diff** for:

   - What the change actually does vs. what the PR description/title claims
   - Correctness concerns or obvious bugs
   - Scope creep (unrelated changes bundled in)
   - Whether it follows this repo's conventions (see `CLAUDE.md` if present — e.g. `data/` should
     never be committed, `requirements.txt` should only grow when a dependency is truly needed)
   - Whether the PR template sections (type of change, test plan) were filled in meaningfully,
     or left as placeholders
   - Unresolved review comments or failing status checks

4. **Produce a report**, one section per PR, ordered by PR number:

   ```markdown
   ### #<number> — <title> (<author>, <headRefName> → <baseRefName>)
   <url>

   **Summary:** one or two sentences on what this PR does and why.

   **Changes:** key files/areas touched, called out by name.

   **Review notes:** correctness/scope/convention concerns found in the diff, if any —
   otherwise state that none were found.

   **Open items:** unresolved comments, failing checks, or missing test plan — otherwise "none".

   **Assessment:** one of Looks good / Needs changes / Needs discussion, with a one-line reason.
   ```

   End with a short overall summary line (e.g. "3 open PRs: 2 look good, 1 needs changes").

This skill is read-only — it never comments on, approves, merges, or closes a PR. Report findings
back to the user and let them decide next steps.
