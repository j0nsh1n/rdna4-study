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
Left open after the Neon fix round: a SIGKILL between the `manual` write and the `changed-by-us` marker left the card in `manual` and skipped the restore (narrowed on 2026-11-26 via `pin-attempt`, though an unawaited in-flight real `sudo` write racing after `SIGKILL` is not covered or tested on hardware; see below). A killed watchdog's orphaned child can still hold the lock descriptor for a moment.

## Not tested
- The real sysfs, real `sudo`, the descriptor-based password step, and `sudo`'s ticket behaviour in a `setsid` child.
- Whether the driver keeps soft limits after `auto` (waits for the card). On a real card `pp_dpm_mclk` marks the current clock
  with `*`, not the mask, so on real sysfs (no `.mclk-mask` sidecar) the pre-state is accepted only when `power_dpm_force_performance_level` is `auto` and `pp_dpm_mclk` lists at least two levels; `manual`, a single level or an unreadable file fails closed. The dry run models the mask in a sidecar
  file. The restore read-back therefore cannot prove the mask on real hardware: measure it on the card first.
- The hotspot guard of design point 6 (only the time cap is built; the gate's 95 C stop exists already).
- `RESTORE-FAILED` is never created in the real fixed place by any test.

## How to run
`scripts/round10/selftest-p0.sh` (CPU only, about 70 s; scratch under `runs/<UTC>-r10p0/`).
Dry driver: `R10_SYSFS=<dir with the three files> R10_LOCK=<scratch lock> scripts/round10/pin_mclk.sh --dry-run --level 3 -- <cmd>`.
Real run (owner only, after the card is installed): also needs `R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved` and
`R10_SECRET_FILE`; run the no-op pin (level 1) first.

## 2026-11-26 dry-run hardening updates

`scripts/round10/selftest-p0.sh` now contains 42 success lines (`ok "..."`), and passed 42 PASS / 0 FAIL in the 2026-11-26 dry-run verification. The "22 PASS" figure in the Neon fix round above is that earlier run, not this count. The last of the 42 is `sudo was never called in any case`. Real sysfs and real `sudo` are still not run. The stub `sudo` is the stand-in the script checks for, and that last line is how the script records that the stand-in was never called.

The 42 lines include the earlier pass, break, cap, and guard cases and the four Neon fixes. The later cases, each with a copy that drops the new check so the test can see it missing:

1. Hardcoded lock. `follow-up 1`: a dry run whose `R10_LOCK` is the checkout default lock (`STUDY_GPU_LOCK` elsewhere) is refused before `flock` (exit 2, no run folder).
2. Guard before run-folder writes. `follow-up 2`: `__locked` refused by the guard writes nothing into the run folder (no `job.pid`, `events.log`, or `locked`).
3. Process state. `follow-up 3a`: `restore.rc` is replaced by rename, and a symlink is not written through. `follow-up 3b`: `alive()` / `claim_alive()` are right for a comm field `a b) Z c` (live, start time matches), a zombie with comm `a b) c`, a reused-pid start time, and a missing pid. `follow-up 3c`: a claim that names a live process with a different start time is taken over at once. The copy without the start-time check keeps waiting on the reused pid.
4. `R10_CRED_FD`. `follow-up 4`: an inherited `R10_CRED_FD` with a command in it is ignored on a dry run without a secret file. No command runs, the payload runs, the variable is unset, and no credential descriptor is logged. The copy with neither the unset nor the digit check runs the injected command.

Outside-review cases in the same script:

5. `kimi 5`: a second run in the same folder is refused before the lock (exit 2, events unchanged, level `auto`, lock free). The copy that reuses the folder does not refuse.
6. `kimi 6`: a real `pp_dpm_mclk` listing, level `auto`, six levels, no sidecar, is accepted and restored. Level `manual` fails closed (exit 20) and is not reported as mask `all`. An unreadable performance level fails closed. The copy of the old mask check reports `all` for a manual level with six listed levels. The real card still uses the rule already stated above: with no sidecar the pre-state is accepted only when the level is `auto` and at least two levels are listed (the live file marks the current clock with `*` rather than a mask); every other state, including `manual`, a single level and an unreadable file, fails closed.
7. `kimi 7`: `__locked` with the owner switch but without lock descriptor 9 exits 2 before any run file, and the stub `sudo` is not called. The copy without that check writes a run file or calls `sudo`.
8. `kimi R2 1`: the same inode is not proof of the lock. `lock_fd9_ok` now also reads `/proc/$$/fdinfo/9` and requires a `lock: N: FLOCK ADVISORY WRITE pid maj:min:ino` line with the lock file's inode. On this kernel that line appears only for an open file description holding an exclusive flock (also when inherited by the `__locked` child); an open-only descriptor, a descriptor beside a competing holder, and a shared lock (`READ`) show none. Two cases: `1a` fd 9 opened on the lock without `flock`, `1b` a background process holds the flock while fd 9 is opened without it. Both exit 2 before any run file and without calling `sudo`. A copy with an inode-only check lets both through. The self-test now ends with 42 PASS, 0 FAIL, and the stub `sudo` is never called.

## 2026-11-26 real hardware `sudo` validation (idle payloads, MEASURED)

With explicit owner approval (`R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved` and one-use `R10_SECRET_FILE` delivered via private secret card, unlinked by `cred_open`), `scripts/round10/pin_mclk.sh` ran twice on the real RX 9070 XT (`0x1002:0x7550`, LACT track `overclock`, monitors on, no GPU compute load). Raw logs are in the companion checkout's git-ignored build tree (`afk-r13/pin-mclk-level1-test/` and `afk-r13/pin-mclk-level5-test/`).

1. **Level 1 no-op test (`--level 1`, 2 s idle payload):**
   - Pre-state: `auto`, `pp_dpm_mclk` listed 6 levels (`0: 96Mhz`, `1: 456Mhz *`, `2: 772Mhz`, `3: 875Mhz`, `4: 1124Mhz`, `5: 1258Mhz`).
   - Pin: `write 'manual' to power_dpm_force_performance_level: status 0`, `write '1' to pp_dpm_mclk: status 0`. Inside the payload, sysfs read `manual` and `1: 456Mhz *`.
   - Restore: `write '0 1 2 3 4 5' to pp_dpm_mclk: status 0`, `write 'auto' to power_dpm_force_performance_level: status 0`, `RESTORE-OK level=auto mask=all`, exit 0, no `RESTORE-FAILED`, lock free, 0 amdgpu faults in `journalctl -k`.
2. **Level 5 idle test (`--level 5`, 3 s `r11_sampler` at 100 ms period):**
   - Pin: `write 'manual' ... status 0`, `write '5' to pp_dpm_mclk: status 0`.
   - Observed idle clock under `--level 5`: inside the payload, `pp_dpm_mclk` read **`4: 1124Mhz *`** (not `5: 1258Mhz *`), and `r11_sampler` reported a flat **`uclk_cur_MHz = 1124`** across all 30 samples (`gfxclk_cur_MHz` 195–238 MHz, hotspot 43–44 C, power 40 W).
   - Restore: `write '0 1 2 3 4 5' ... status 0`, `write 'auto' ... status 0`, `RESTORE-OK level=auto mask=all`, exit 0, returned immediately to `auto` / `1: 456Mhz *`, no `RESTORE-FAILED`, lock free, 0 amdgpu faults.

**Open after idle validation:** Watchdog crash-recovery (`SIGKILL`) on real `sudo` was not triggered on hardware.

## 2026-11-26 real hardware under-load validation, `listing()` fix, and `pin-attempt` kill-window narrowing (MEASURED)

Three real GPU runs exercised `scripts/round10/pin_mclk.sh --level 5 --cap-s 2400` on the RX 9070 XT (`1002:7550`, LACT track `overclock`, monitors on; raw logs under `kimi-next/pinned-slot-{1,2,3}/` in the companion checkout's git-ignored build tree):

1. **Real-hardware `listing()` false-positive `RESTORE-FAILED` bug caught (`pinned-slot-1`) and fixed:**
   - In `pinned-slot-1` (18 s under load; later superseded for measurement tables by `pinned-slot-3` because Cyberpunk 2077 ran during part of slot 1), both restore writes succeeded (`write '0 1 2 3 4 5' ... status 0`, `write 'auto' ... status 0`) and sysfs read back `level=auto` with all 6 levels present, yet `pin_mclk.sh` exited `21` and wrote `RESTORE-FAILED`.
   - Byte inspection (`cat -A`) proved the cause: `amdgpu` appends a trailing space to every inactive `pp_dpm_mclk` line (`"1: 456Mhz "`) and no trailing space after `*` on the active line (`"1: 456Mhz *"`). The old `listing()` filter (`sed 's/[[:space:]]*\*[[:space:]]*$//'`) stripped the space only on the `*` line, so when the active `*` moved from level `1` before the run to level `3` (`3: 875Mhz *`) immediately after `auto` restore, `before.mclk.norm` and live `listing()` differed by trailing spaces.
   - Fix: `listing()` in `scripts/round10/pin_mclk.sh` now uses `sed -E 's/[[:space:]]*\*?[[:space:]]*$//'`.
2. **`SIGKILL` kill window narrowed with `pin-attempt` (following outside review on `933b9e4`):**
   - `pin_mclk.sh` now creates `$R10_RUN/pin-attempt` *before* the first sysfs write (`manual`). In `do_restore()`, when `changed-by-us` is absent and `pin-attempt` exists: if the job itself is cleaning up (`R10_WHO != watchdog`) and sysfs still reads `auto`/`all`, it logs `RESTORE-SKIPPED pin attempted but state still auto/all` without writing sysfs; if the watchdog is recovering from a killed parent (`R10_WHO == watchdog`), it always restores `0 1 2 3 4 5` and `auto`.
   - Limitation kept explicit: on real hardware `sysfs_write` runs `sudo -n bash -c ...` as a child process, so if `SIGKILL` hits while that `sudo` child is still in flight, the orphan child could still race with the watchdog's restore. `SIGKILL` during real `sudo` writes remains untested on hardware.
   - On real hardware in `pinned-slot-2` (where the sudo credential expired before the first write), `pin_mclk.sh` exited `23` cleanly in 5 s with `RESTORE-SKIPPED pin attempted but state still auto/all`, `restore status 0`, no `RESTORE-FAILED`, lock free, and 0 amdgpu faults.
3. **Clean under-load validation and real-hardware `RESTORE-OK` (`pinned-slot-3`):**
   - With Cyberpunk closed, `pinned-slot-3` ran 25 s under `pin_mclk.sh --level 5 --cap-s 2400` across 8 `ld_mat2` jobs (one-wave and whole-card FP8 WMMA + `MALL`/`GDDR6` up to `3.24 GHz` SCLK, `288 W` reported power, `65 C` hotspot) plus 4 Item 3 Step 1 Vulkan/HIP jobs.
   - `uclk_cur_MHz` held flat at **`1124 MHz`** (`4: 1124Mhz *`) across all **1,247** `r11_sampler` rows (**108** `ld_mat2` + **1,139** Item 3 watch rows; distinct `uclk = [1124]`) and every `uclk_start`/`uclk_end` row of Item 3 (`1124 -> 1124`).
   - At restore, both writes succeeded (`status 0`) and the fixed driver logged **`job RESTORE-OK level=auto mask=all`** (`restore status 0`), with no `RESTORE-FAILED`, lock free, and 0 amdgpu faults in `journalctl -k`. Note that on real `amdgpu` sysfs (no `.mclk-mask` sidecar), `mask=all` reports `level=auto` with at least two listed levels; `pp_dpm_mclk` does not independently expose the driver's manual bitmask.
4. **Dry-run self-test suite (`selftest-p0.sh`):**
   - Added 5 regression cases (`hw-star-move`, `hw-star-move break`, `pin-attempt-window`, `pin-attempt-window break`, and `pin-attempt-skip` using dry-only hooks `R10_TEST_PAUSE_BEFORE_MARK` and `R10_INJECT=pin-write`) and suppressed transient `/proc/<pid>/cmdline` stderr on vanished processes.
   - Verified **47 PASS / 0 FAIL / 0 stderr lines** (`sudo was never called in any case`).
