# Round 10 P0: a safe restore path for memory-clock pinning (dry run)

## Plain summary
`scripts/round10/pin_mclk.sh` pins the memory clock for one payload and puts the card back (level `auto`, full mask) while the
shared lock is still held. It was proven only as a dry run: plain files stand in for sysfs, a scratch lock stands in for the real
one, a stub `sudo` would have recorded any call (none happened), and no GPU was used. The old problem (restore after the lock was
released) is gone in the dry run: killing the job with SIGKILL leaves the lock held until the watchdog has restored, and the
test shows it can see the problem when the watchdog is started without the lock. The real sysfs, real `sudo` and the real
card are NOT tested.

## What changed
- New `scripts/round10/pin_mclk.sh` (driver, with the watchdog inside it) and `scripts/round10/selftest-p0.sh`.
- `scripts/round9/common.sh`: `gate()` refuses (exit 9, counted as a failed gate) while `RESTORE-FAILED` exists; checked before
  and after the HOLD wait. Fixed place: `runs/RESTORE-FAILED` in the checkout (override `R9_RESTORE_FAILED`
  for the gate, `R10_RESTORE_FAILED` for the driver; the driver also honours `R9_RESTORE_FAILED`).
- `scripts/round9/selftest_driver.sh` covers the refusal and a control (job runs once the file is gone). `stable_clocks.sh` and
  its `G1_REDESIGNED` guard are untouched.

## Design as built
1. The outer script only checks guards, makes the run folder and takes the lock (`flock` re-entry, like `study-llm.sh`). Every
   sysfs read-for-decision and write is in the locked job.
2. Pre-state: level must read `auto` and the mask must allow all levels, else exit 20 "NOT CHANGED: level was X", nothing written,
   no watchdog started. `changed-by-us` is created only after the first successful write (`manual`).
3. Watchdog: `setsid`, inherits the lock descriptor, polls the parent (`kill -0`, zombie counts as dead) and a `done` token.
   Parent dead: kill the payload's process group, restore if `changed-by-us` exists. The payload runs without the lock and
   credential descriptors, so only the job and the watchdog hold the lock.
4. Restore: mask `0 1 2 3 4 5` (indices of the before-state listing), then `auto`, then read both back and compare with the
   before-state. A restore is claimed once (`mkdir restore-claim`), so the job and the watchdog never both write. Failure writes
   `RESTORE-FAILED`, exit 21.
5. Credentials: one function, `credential_step` (stubbed in dry runs); `cred_open` opens the password file as a descriptor and
   unlinks it; the real step is `sudo -S -p '' -v < /proc/self/fd/N`. **Written, never run.**
6. Cap: `R10_CAP_S` (default 120, `--cap-s`); the watchdog kills the payload and does the restore itself; exit 22.
7. `R10_SYSFS` is required; the directory's `device` file must read `0x7550` (checked in dry runs too). A real run refuses
   without `R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved`. That switch is documented in the script header and was never set.
   A dry run also refuses the real lock path and any path under `/sys`.

## What each case showed (3 runs in a row, 15 PASS each, exit 0)
- **pass**: exit 0, level pinned to index 3 then back; level `auto`, mask `0 1 2 3 4 5`, no `RESTORE-FAILED`.
- **break 1**: job SIGKILLed after pinning; the poller got the lock 16 ms after the watchdog's restore line.
  ```
  ...323.658 job pinned: level manual, mclk index 1
  ...323.721 watchdog restoring (parent alive: no)
  ...323.925 watchdog write '0 1 2 3 4 5' to pp_dpm_mclk
  ...324.077 watchdog write 'auto' to power_dpm_force_performance_level
  ...324.080 watchdog RESTORE-OK level=auto mask=all      (lock acquired by the poller at ...324.096)
  ```
- **break 1b**: watchdog started without the lock descriptor; the lock was acquired about 0.4 s BEFORE the restore line, so the
  same check fails, as it must (the test reports this as PASS: "the gap is visible").
- **break 2a / 2b**: level `manual`, or mask already limited: exit 20, no write, no restore, state unchanged.
- **break 3**: injected restore-write failure: exit 21, `RESTORE-FAILED` exists, `gate()` refuses (exit 9) and does not start its
  job; control: with the file removed the same call starts the job.
- **break 4**: watchdog has no credential (job SIGKILLed): watchdog exit 1, `RESTORE-FAILED` written
  (`watchdog RESTORE-FAILED level now 'manual', mask 'limited: 1'`), `gate()` refuses.
- **break 5**: second pinned job started while the first held the lock: its locked start came 60 ms after the first job's restore;
  it read `auto`; both exit 0.
- **cap**: payload over 1 s killed by the watchdog, which restored; exit 22. (A first version let the job win the restore race;
  fixed so the watchdog restores after a cap kill.)
- **guards**: no switch: refused, exit 2. Device `0x744c`: refused, exit 2. Copies with each guard line deleted do not refuse
  (the test can see a missing guard). The stub `sudo` was never called.
- Old self-tests: `selftest_driver.sh` 7 PASS before; 9 PASS after (the two new gate cases). `selftest_g1.sh` all PASS before and
  after. Before-output is in `runs/old-selftests-before.txt` (untracked).

## Fix round after the outside review (Neon, FAIL with 4 blocking, on 6eb2018)
Each fix has a self-test case (22 PASS, 0 FAIL, three runs in a row). Each new case FAILS against `git show 6eb2018:...pin_mclk.sh`
(run with `R10_SELFTEST_DRIVER=<old copy>`; the old driver also hung on the direct `__watchdog 1` call, so the new calls use `timeout`).
1. Entry points: `guard()` (device 0x7550; real run needs the switch; dry run refuses `/sys` and both real lock paths) is called at the
   top of `locked_job`, `watchdog`, `cred_open`, `credential_step` and `sysfs_write`. Case "fix 1": `__locked`/`__watchdog` with no switch exit 2,
   stub `sudo` not called, secret file kept. On old code the secret file was unlinked and the watchdog call never returned.
2. A killed claimer: the claim is a directory that appears already holding its owner's pid (rename into place). A waiter whose claimer
   is dead and has no `restore.rc` takes over; if the claimer is alive after 120 s it writes `RESTORE-FAILED`. Cases "fix 2a" (job
   SIGKILLed inside `do_restore`) and "fix 2b" (watchdog SIGKILLed after claiming during a cap kill): level `auto`, lock freed after the restore.
   Old code 2a ended with the level `manual`, no `RESTORE-FAILED`. Test hook: `R10_TEST_RESTORE_DELAY`/`_WHO` (dry run only).
3. SIGTERM during the restore: INT/TERM/HUP are ignored from the start of `finish_restore` and in the watchdog. Case "fix 3": restore.rc 0, one RESTORE-OK, no
   deadlock. Old code: exit 21 after a 31 s deadlock, level left `manual`.
4. `R10_CRED_FD` is exported by `cred_open` and unset for the payload. Case "fix 4" (dry run, stub secret file, content never printed): the
   watchdog sees the same descriptor number and reads `/proc/self/fd/N` from offset 0 twice; the payload has neither variable nor descriptor.
   Dry-only hooks: `R10_TEST_SECRET_FILE`, `R10_TEST_CRED_PROBE`.
Follow-ups done: `break 1b` fails when its timestamps are missing; the driver refuses (exit 2) while `RESTORE-FAILED` exists (outer and in the locked job).
Left open: a SIGKILL between the `manual` write and the `changed-by-us` marker leaves the card in `manual` and the restore skipped
(a `pin-attempt` marker written before the first write would close it). A killed watchdog's orphaned child can hold the lock descriptor for a moment.

## Not tested
- The real sysfs, real `sudo`, the descriptor-based password step, and `sudo`'s ticket behaviour in a `setsid` child.
- Whether the driver keeps soft limits after `auto` (waits for the card). On a real card `pp_dpm_mclk` marks the current clock
  with `*`, not the mask, so the real mask check is only "at least 2 levels listed"; the dry run models the mask in a sidecar
  file. The restore read-back therefore cannot prove the mask on real hardware: measure it on the card first.
- The hotspot guard of design point 6 (only the time cap is built; the gate's 95 C stop exists already).
- `RESTORE-FAILED` is never created in the real fixed place by any test.

## How to run
`scripts/round10/selftest-p0.sh` (CPU only, about 70 s; scratch under `runs/<UTC>-r10p0/`).
Dry driver: `R10_SYSFS=<dir with the three files> R10_LOCK=<scratch lock> scripts/round10/pin_mclk.sh --dry-run --level 3 -- <cmd>`.
Real run (owner only, after the card is installed): also needs `R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved` and
`R10_SECRET_FILE`; run the no-op pin (level 1) first.
