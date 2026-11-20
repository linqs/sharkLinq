# Shared Matching Demo

A disposable matching exercise, available as **Demo** in the mode selector. All participants share one demo database, copied from the application database when the server starts.

## Start

Needs a populated application database, its images, and a running inference service (see the app's `README.md`).

```bash
python3 -m scripts.shark_identification --run --demo
```

**Demo** now appears in the mode selector. Without `--demo` it is missing.

## Play a Round

1. Choose **Demo** in the mode selector.
2. Expand the **Demo** menu.
3. Choose a count and press **Randomize**.
4. Press **Browse randomized sharks** to find the detached images.
5. Match them: a merge commits at once and reports whether it was correct.

Viewers can browse. Collaborators and higher can also randomize, unmatch, merge, and clear.

- Each selected image becomes its own shark, labelled `unmatched` and `randomized`.
- A profile's **Match Key** shows the original shark; **Hide** conceals it again.
- Sampling is deterministic: the same count gives the same selection.
- At least one image always stays with the original shark. An invalid count changes nothing.
- A merge is correct only if it rejoins the original shark without adding another.
- Wrong merges never reveal the answer.

## Reset

| Action | Effect |
| --- | --- |
| **Randomize** | Discards all demo changes, then samples a new set |
| **Clear randomized images** | Discards all demo changes, no new set |
| Leave Demo | Nothing |
| Restart the server | Discards progress, captures a fresh baseline |

Other open demo pages must reload after a reset; stale requests get a conflict response.

## Limits

- Not available: uploads, deletions, profile and image edits, embedding updates, proposals, account changes.
- The limits apply to the Demo mode only; a signed-in admin or owner in another mode keeps the regular permissions.
- Unavailable tabs are disabled but visible.
- Demo operations create no Actions records.

## Developers

- `runtime.py`: baseline, sampling, grading.
- `api.py`: authorization, restrictions, database scoping.
- `demo.js`: browser menu.
- Demo requests carry `X-Shark-Demo: true` and the current `X-Shark-Demo-Round`; without the first header, `demo/` endpoints answer `404 Demo mode is unavailable.` before checking the session.
- Tests: `python3 -m pytest -q tests/demo/test_demo.py`
