<!-- Spec: unit-test work ledger for the I12/I13 + S14 slice · operator request 2026-08-31
     ("log the unit test work to playbook-unit-tests.md"). Companion to playbook.md. -->

# Playbook — unit tests for the structural-evidence / one-call / bridge slice

Why this file exists: the transport signature widened (`_via_openrouter` gained
`images` and `max_tokens`), which broke every monkeypatched fake backend in
tests/test_interpret.py that pinned the old 3-arg shape. Fix cycles were slow
because `reason()`'s new default `structure="image"` made each fake-backend
test render a matplotlib figure it never inspects. Both causes are fixed in
the working tree; what is verified vs pending is logged here.

## Layer 1 — fixes applied (in working tree, this commit)

- [DONE] T1 widen `_one_shot_backend` fake to the new transport signature
  _Files:_ tests/test_interpret.py:561  _Verify:_ pytest tests/test_interpret.py -q -k one_shot
  _Lessons:_ a monkeypatched fake is a CONTRACT PIN on the transport signature;
  widening the real function without grepping for fakes broke 3 tests. Next
  signature change: `grep -n "def be(" tests/` first.
- [DONE] T2 widen the inline staged fake the same way
  _Files:_ tests/test_interpret.py:516  _Verify:_ pytest tests/test_interpret.py -q -k staged
- [DONE] T3 `structure="none"` on all 7 fake-backend `reason()` calls — a fake
  test must not pay for an image render it never reads (was ~50 s/test)
  _Files:_ tests/test_interpret.py  _Verify:_ time pytest tests/test_interpret.py -q -k "one_shot" (< 30 s)

## Layer 2 — verification pending (run after restart; sequential)

- [OPEN] T4 full interpret suite green
  _Files:_ tests/test_interpret.py  _Verify:_ pytest tests/test_interpret.py -q
  _Note:_ last full run (pre-T2/T3) was 3 failed / 150 passed; the 3 were the
  fake-signature breaks. Expect green now. ~5 min: several tests exercise the
  LIVE OpenRouter path with 90 s timeouts.
- [OPEN] T5 whole-repo suite + commit gate
  _Verify:_ set -o pipefail; python -m pytest -q
  _Note:_ fast suites already verified green post-change: test_walker_render +
  test_sampler + test_graph_tools = 118 passed; new pins s14/w16 = 2 passed,
  i12/i13 = 4 passed, one_shot = 2 passed 1 skipped.

## Layer 3 — test debt worth paying (parallel, any time)

- [OPEN] T6 mark the live-transport tests `@pytest.mark.live` so the default
  run is fake-only and finishes in ~1 min; live lane runs on demand
  _Files:_ tests/test_interpret.py  _Verify:_ pytest -q -m "not live" (< 90 s)
- [OPEN] T7 one shared fake-backend factory instead of three hand-rolled fakes,
  so the next transport change touches one place
  _Files:_ tests/test_interpret.py  _Verify:_ pytest tests/test_interpret.py -q
