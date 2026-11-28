#!/usr/bin/env bash
# Round 11 item 4a: the video engine (VCN) through VA-API: decode and encode of H.264, HEVC and AV1 with ffmpeg, with power and clocks beside it.
#
# Usage: run_r11_video.sh [--plan] [--cpu-checks] [--make-inputs] [--cpu-reference] [--run] [--smoke] [--break-device] [--out DIR]
#   --plan           print the cells, the controls, the guards and the estimated card time; no lock, no GPU
#   --cpu-checks     no GPU, no card access beyond sysfs reads: bash -n, tools and VA-API wrappers present in ffmpeg (listed, the device is NOT opened),
#                    the render node of the dGPU found by device id (0x7550) and the iGPU's node (0x13c0) told apart, the controls (frame md5, PSNR floor,
#                    video-activity check) on tiny CPU-made clips, each with its deliberate break, the gate's self-test
#   --make-inputs    (CPU only, nice 19, owner not gaming) the synthetic raw inputs and the CPU-encoded clips, in R11_VIDEO_SCRATCH
#   --cpu-reference  (CPU only) CPU-encoder PSNR of each source at 8 and 40 Mbit/s: the number that fixes the PSNR floor of the hardware encodes
#   --run            the card jobs: 60 s idle baseline, decode cells, encode cells, 60 s idle after; one locked window, both gates; --smoke runs 1 cell of each
#   --break-device   with --run: ALSO decode one clip on the iGPU's render node; the dGPU activity check must FAIL (deliberate break 1)
# Environment: R11_VIDEO_SCRATCH (inputs, default <repo>/runs/r11-video-scratch: the home disk, never /tmp), R11_CELL_S (target seconds per cell, default 6),
#   R11_PSNR_FLOOR (dB, default 26, the value fixed in expect-4a.md from --cpu-reference), R11_DRY=1 (print the ffmpeg commands instead of running them),
#   R11_FRAMES (default 300), plus everything in r11_common.sh. The render node is never given by the user: it is found by device id.
# Nothing here writes sysfs, uses sudo, or changes git.
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
# shellcheck source=r11_common.sh
. "$here/r11_common.sh"
SCR=${R11_VIDEO_SCRATCH:-$R11_REPO/runs/r11-video-scratch}
FRAMES=${R11_FRAMES:-300}
CELL_S=${R11_CELL_S:-6}
PSNR_FLOOR=${R11_PSNR_FLOOR:-26}
PLAN=0 CPU=0 MAKE=0 CPUREF=0 RUN=0 SMOKE=0 BREAKDEV=0 INNER="" OUT=${R11_RUN:-}
while (($#)); do
    case $1 in
        --plan) PLAN=1 ;; --cpu-checks) CPU=1 ;; --make-inputs) MAKE=1 ;; --cpu-reference) CPUREF=1 ;; --run) RUN=1 ;; --smoke) SMOKE=1 ;;
        --break-device) BREAKDEV=1 ;; --out) OUT=${2:-}; shift ;; --inner) INNER=${2:-}; shift ;;
        *) echo "unknown option $1 (see the header of $SELF)"; exit 2 ;;
    esac
    shift
done
RES_NAMES=(1080p 4k)
declare -A RES_W=([1080p]=1920 [4k]=3840) RES_H=([1080p]=1080 [4k]=2160)
CODECS=(h264 hevc av1)
declare -A SWENC=([h264]=libx264 [hevc]=libx265 [av1]=libsvtav1) HWENC=([h264]=h264_vaapi [hevc]=hevc_vaapi [av1]=av1_vaapi)
declare -A EXT=([h264]=mp4 [hevc]=mp4 [av1]=mkv)

mono_ns() { perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%d\n", clock_gettime(CLOCK_MONOTONIC)*1e9'; }

# ---- cells ----------------------------------------------------------------------------------------------------------
# decode: codec x res x output (hw surface kept on the card / copied to system memory) x parallel processes 1,2,3
# encode: codec x (1080p 8M, 1080p 40M, 4k 40M) x quality (h264 only has -quality: default and 0) x parallel 1,2,3
cells() { # prints "kind codec res out_or_br quality n"
    local c r o n q br
    for c in "${CODECS[@]}"; do for r in "${RES_NAMES[@]}"; do for o in hw sys; do for n in 1 2 3; do echo "dec $c $r $o - $n"; done; done; done; done
    for c in "${CODECS[@]}"; do
        for rb in 1080p:8 1080p:40 4k:40; do
            r=${rb%%:*}; br=${rb##*:}
            for q in default 0; do
                [[ $c != h264 && $q != default ]] && continue
                for n in 1 2 3; do echo "enc $c $r ${br}M $q $n"; done
            done
        done
    done
}
ncells() { cells | wc -l; }

do_plan() {
    local n
    n=$(ncells)
    echo "PLAN r11 video (item 4a): $n cells at about ${CELL_S} s each = about $((n * CELL_S / 60 + 1)) min of video-engine load, plus 2 x 60 s idle baseline: about 8-10 min card time"
    echo "PLAN decode: ffmpeg -hwaccel vaapi on the dGPU render node (found by device id 0x7550), hardware surface kept vs copied to system memory, 1/2/3 parallel processes (do two engines double the rate?)"
    echo "PLAN encode: h264_vaapi hevc_vaapi av1_vaapi, 1080p 8 and 40 Mbit/s, 4K 40 Mbit/s, h264 also at -quality 0, 1/2/3 parallel"
    echo "PLAN controls: decode frame md5 equals the CPU decoder's for every frame; encode output decodes on the CPU with the right frame count and size and PSNR >= floor ($PSNR_FLOOR dB until --cpu-reference fixes it);"
    echo "PLAN           the dGPU's vclk0/dclk0/mm activity in the cell's window must exceed its idle baseline (the iGPU did the work otherwise: cell void)"
    echo "PLAN breaks: (1) decode on the iGPU node: the activity check must FAIL (--break-device); (2) one byte of a stream corrupted: the md5 comparison must FAIL; (3) PSNR against a different clip: must FAIL"
    echo "PLAN guards: study-style gate (lock, nice -n 10, 8 GiB, hotspot <= ${R11_START_C} C to start, ${R11_GUARD_C} C guard, journalctl -k fault check); sampler r11_sampler at 100 ms beside every job"
    echo "PLAN host needs: memory under 4 GB (4K frames), scratch about 8 GB (raw inputs plus clips) in $SCR"
    echo "PLAN no ffmpeg call opens the card in --plan, --cpu-checks or --make-inputs; --run opens the dGPU render node only"
}

# ---- the controls (pure functions of files, so the CPU checks can break them) ---------------------------------------------------
# md5_equal <a.framemd5> <b.framemd5>: 0 when every frame line matches (comment lines ignored).
md5_equal() { diff -q <(grep -v '^#' "$1") <(grep -v '^#' "$2") >/dev/null; }
# psnr_of <decoded clip> <raw source> <WxH>: average PSNR (dB) of the Y plane per ffmpeg's psnr filter, or empty.
psnr_of() {
    ffmpeg -hide_banner -nostdin -loglevel info -i "$1" -f rawvideo -pix_fmt yuv420p -s "$3" -r 30 -i "$2" -lavfi "[0:v]format=yuv420p[a];[a][1:v]psnr" -f null - 2>&1 |
        sed -n 's/.*PSNR y:\([0-9.]*\).*/\1/p' | tail -1
}
psnr_ok() { awk -v p="$1" -v f="$2" 'BEGIN { exit !(p != "" && p+0 >= f+0) }'; }
# stream_ok <clip> <frames> <WxH>: the clip decodes on the CPU to the right number of frames at the right size.
stream_ok() {
    local n s
    n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$1" 2>/dev/null)
    s=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$1" 2>/dev/null)
    [[ $n == "$2" && $s == "$3" ]]
}
# activity <samples.csv> <t0_ns> <t1_ns>: "max_vclk0 max_dclk0 max_vclk1 max_dclk1 max_mm_act rows" within the window (r11_sampler columns 16-20).
activity() {
    awk -F, -v a="$2" -v b="$3" '!/^#/ && $1+0 >= a+0 && $1+0 <= b+0 && $NF == 1 { n++; for (i = 16; i <= 20; i++) if ($i+0 > m[i]) m[i] = $i+0 } END { printf "%d %d %d %d %d %d\n", m[16], m[17], m[18], m[19], m[20], n }' "$1"
}
# activity_ok <samples.csv> <t0> <t1> <idle line from activity>: video block busier than its idle baseline in at least one field, with rows in the window.
activity_ok() {
    local cur idle
    cur=$(activity "$1" "$2" "$3")
    idle=$4
    awk -v c="$cur" -v i="$idle" 'BEGIN { split(c, C, " "); split(i, I, " "); ok = 0; for (k = 1; k <= 5; k++) if (C[k] > I[k] && C[k] != 65535) ok = 1; exit !(ok && C[6] > 0) }'
}

# ---- render nodes ------------------------------------------------------------------------------------------------------------
render_for_id() { # device id -> /dev/dri/renderDxxx (empty when none)
    local r
    for r in /sys/class/drm/renderD*; do [[ $(cat "$r/device/device" 2>/dev/null) == "$1" ]] && { echo "/dev/dri/$(basename "$r")"; return 0; }; done
    return 1
}

# ---- inputs ---------------------------------------------------------------------------------------------------------------------
raw_of() { echo "$SCR/raw_$1.yuv"; }
clip_of() { echo "$SCR/clip_$1_$2.${EXT[$1]}"; }
make_raw() { # res frames dir
    local r=$1 f=$2 d=$3 w=${RES_W[$1]} h=${RES_H[$1]}
    nice -n 19 ffmpeg -hide_banner -nostdin -y -loglevel error -f lavfi -i "testsrc2=size=${w}x${h}:rate=30,noise=alls=8:allf=t" -frames:v "$f" -pix_fmt yuv420p -f rawvideo "$d/raw_$r.yuv"
}
cpu_encode() { # codec raw WxH frames out [bitrate]
    local c=$1 raw=$2 sz=$3 f=$4 out=$5 br=${6:-}
    local args=()
    [[ -n $br ]] && args=(-b:v "$br")
    case $c in h264) args+=(-preset veryfast) ;; hevc) args+=(-preset veryfast -x265-params log-level=error) ;; av1) args+=(-preset 10) ;; esac
    nice -n 19 ffmpeg -hide_banner -nostdin -y -loglevel error -f rawvideo -pix_fmt yuv420p -s "$sz" -r 30 -i "$raw" -frames:v "$f" -c:v "${SWENC[$c]}" "${args[@]}" -pix_fmt yuv420p "$out"
}
cpu_framemd5() { nice -n 19 ffmpeg -hide_banner -nostdin -y -loglevel error -i "$1" -pix_fmt yuv420p -f framemd5 "$2"; }

do_make_inputs() {
    mkdir -p "$SCR"
    local free_kb
    free_kb=$(df -Pk "$SCR" | awk 'NR==2 {print $4}')
    ((free_kb > 20 * 1024 * 1024)) || r11_die 2 "less than 20 GB free in $SCR"
    local r c
    for r in "${RES_NAMES[@]}"; do
        [[ -s $(raw_of "$r") ]] || make_raw "$r" "$FRAMES" "$SCR" || r11_die 10 "raw input $r failed"
        for c in "${CODECS[@]}"; do
            [[ -s $(clip_of "$c" "$r") ]] || cpu_encode "$c" "$(raw_of "$r")" "${RES_W[$r]}x${RES_H[$r]}" "$FRAMES" "$(clip_of "$c" "$r")" 20M || r11_die 10 "CPU clip $c $r failed"
            cpu_framemd5 "$(clip_of "$c" "$r")" "$SCR/ref_${c}_$r.framemd5" || r11_die 10 "CPU framemd5 $c $r failed"
        done
    done
    echo "inputs ready in $SCR"
}

do_cpu_reference() { # the PSNR the CPU encoders reach at 8 and 40 Mbit/s: the hardware floor is set a few dB below the lowest
    mkdir -p "$SCR"
    local r c br out p lo=999
    for r in "${RES_NAMES[@]}"; do
        [[ -s $(raw_of "$r") ]] || { echo "no raw input $(raw_of "$r"): run --make-inputs first"; exit 2; }
        for c in "${CODECS[@]}"; do for br in 8M 40M; do
            out=$SCR/cpuref_${c}_${r}_$br.${EXT[$c]}
            cpu_encode "$c" "$(raw_of "$r")" "${RES_W[$r]}x${RES_H[$r]}" "$FRAMES" "$out" "$br" || { echo "cpu encode $c $r $br failed"; continue; }
            p=$(psnr_of "$out" "$(raw_of "$r")" "${RES_W[$r]}x${RES_H[$r]}")
            echo "CPUREF $c $r $br psnr_y_dB=$p"
            awk -v p="$p" -v l="$lo" 'BEGIN { exit !(p != "" && p+0 < l+0) }' && lo=$p
        done; done
    done
    echo "CPUREF lowest PSNR $lo dB: set R11_PSNR_FLOOR a few dB below it and write the value into expect-4a.md"
}

# ---- the card jobs (the ffmpeg command lines; R11_DRY=1 prints them) ---------------------------------------------------------------
run_or_echo() { if [[ ${R11_DRY:-0} == 1 ]]; then echo "DRY $*"; else "$@"; fi; }
dec_cmd() { # codec res out render loops
    local c=$1 r=$2 o=$3 node=$4 loops=$5
    local hw=(-hwaccel vaapi -hwaccel_device "$node")
    [[ $o == hw ]] && hw+=(-hwaccel_output_format vaapi)
    echo ffmpeg -hide_banner -nostdin -y -loglevel error "${hw[@]}" -stream_loop "$loops" -i "$(clip_of "$c" "$r")" -f null -
}
enc_cmd() { # codec res br quality node frames
    local c=$1 r=$2 br=$3 q=$4 node=$5 f=$6 w=${RES_W[$2]} h=${RES_H[$2]} extra=""
    [[ $q != default ]] && extra="-quality $q"
    echo ffmpeg -hide_banner -nostdin -y -loglevel error -vaapi_device "$node" -stream_loop 0 -f rawvideo -pix_fmt yuv420p -s "${w}x${h}" -r 30 -i "$(raw_of "$r")" \
        -frames:v "$f" -vf "format=nv12,hwupload" -c:v "${HWENC[$c]}" -b:v "$br" -maxrate "$br" $extra -f null -
}
# cell_run <label> <n> <frames per process> <command...>: n parallel copies, wall time of the whole group; appends to video.csv with the monotonic
# window for the activity check.
cell_run() {
    local label=$1 n=$2 frames=$3
    shift 3
    local t0 t1 w0 w1 i pids=() rc=0
    t0=$(mono_ns); w0=$EPOCHREALTIME
    if [[ ${R11_DRY:-0} == 1 ]]; then echo "DRY x$n $*"; else
        for ((i = 0; i < n; i++)); do "$@" >/dev/null 2>>"$R11_RUN/ffmpeg.err" & pids+=($!); done
        for i in "${pids[@]}"; do wait "$i" || rc=1; done
    fi
    t1=$(mono_ns); w1=$EPOCHREALTIME
    echo "$label,$n,$frames,$w0,$w1,$t0,$t1,$rc" >>"$R11_RUN/video.csv"
    return $rc
}
# calibrate <codec> <res> <out> <node>: loops so that one process runs about CELL_S seconds (a 2-loop run on one process sets the rate).
declare -A LOOPS
calibrate() {
    local key="$1_$2_$3" t0 t1 fps
    [[ -n ${LOOPS[$key]:-} ]] && return 0
    if [[ ${R11_DRY:-0} == 1 || -n ${R11_LOOPS:-} ]]; then LOOPS[$key]=${R11_LOOPS:-20}; return 0; fi
    t0=$EPOCHREALTIME
    # shellcheck disable=SC2046
    $(dec_cmd "$1" "$2" "$3" "$4" 2) >/dev/null 2>>"$R11_RUN/ffmpeg.err"
    t1=$EPOCHREALTIME
    fps=$(awk -v a="$t0" -v b="$t1" -v f="$FRAMES" 'BEGIN { d = b - a; printf "%d", (d > 0 ? 3 * f / d : 100) }')
    LOOPS[$key]=$(awk -v fps="$fps" -v s="$CELL_S" -v f="$FRAMES" 'BEGIN { l = int(s * fps / f); if (l < 1) l = 1; if (l > 2000) l = 2000; print l }')
    echo "calibrated $key: $fps fps -> loops ${LOOPS[$key]}" | tee -a "$R11_RUN/calibration.txt"
}
# verify_hw <node>: the controls of the hardware paths, one pass each, not timed. Decode: framemd5 (yuv420p) equals the CPU decoder's for every frame.
# Encode: the hardware stream decodes on the CPU with the right frame count and size and PSNR >= floor.
verify_hw() {
    local node=$1 c r out p bad=0
    for r in "${RES_NAMES[@]}"; do for c in "${CODECS[@]}"; do
        out=$R11_RUN/verify_dec_${c}_$r.framemd5
        run_or_echo ffmpeg -hide_banner -nostdin -y -loglevel error -hwaccel vaapi -hwaccel_device "$node" -i "$(clip_of "$c" "$r")" -pix_fmt yuv420p -f framemd5 "$out"
        if [[ ${R11_DRY:-0} == 1 ]]; then continue; fi
        if md5_equal "$out" "$SCR/ref_${c}_$r.framemd5"; then echo "CONTROL decode_md5 $c $r PASS"; else echo "CONTROL decode_md5 $c $r FAIL"; bad=1; fi
        out=$R11_RUN/verify_enc_${c}_$r.${EXT[$c]}
        run_or_echo ffmpeg -hide_banner -nostdin -y -loglevel error -vaapi_device "$node" -f rawvideo -pix_fmt yuv420p -s "${RES_W[$r]}x${RES_H[$r]}" -r 30 \
            -i "$(raw_of "$r")" -frames:v "$FRAMES" -vf "format=nv12,hwupload" -c:v "${HWENC[$c]}" -b:v 8M "$out"
        p=$(psnr_of "$out" "$(raw_of "$r")" "${RES_W[$r]}x${RES_H[$r]}")
        if stream_ok "$out" "$FRAMES" "${RES_W[$r]}x${RES_H[$r]}" && psnr_ok "$p" "$PSNR_FLOOR"; then echo "CONTROL encode_check $c $r PASS psnr_y_dB=$p floor=$PSNR_FLOOR"; else echo "CONTROL encode_check $c $r FAIL psnr_y_dB=${p:-none} floor=$PSNR_FLOOR"; bad=1; fi
    done; done
    return $bad
}
inner_run() { # the locked window
    local igpu node
    node=$(render_for_id "$R11_DEVID") || exit 3
    igpu=$(render_for_id 0x13c0) || igpu=""
    mkdir -p "$R11_RUN"
    echo "label,n,frames_per_proc,wall_start,wall_end,mono_start_ns,mono_end_ns,failed" >"$R11_RUN/video.csv"
    echo "dGPU render node $node, iGPU render node ${igpu:-none}" | tee "$R11_RUN/nodes.txt"
    [[ $node != "$igpu" ]] || exit 3
    local t0 t1
    t0=$(mono_ns); [[ ${R11_DRY:-0} == 1 ]] || sleep "${R11_IDLE_S:-60}"; t1=$(mono_ns)
    echo "idle-before,0,0,0,0,$t0,$t1,0" >>"$R11_RUN/video.csv"
    local ctl_bad=0
    verify_hw "$node" >"$R11_RUN/controls.txt" 2>&1 || ctl_bad=1  # no pipe: a pipe would return tee's status and hide a failed control
    cat "$R11_RUN/controls.txt"
    if ((ctl_bad)); then echo "a hardware control failed: no timing cell is read"; exit 1; fi
    local kind c r a q n label list
    if ((SMOKE)); then list=$(printf '%s\n' "dec h264 1080p hw - 1" "enc h264 1080p 8M default 1"); else list=$(cells); fi
    while read -r kind c r a q n; do
        if [[ $kind == dec ]]; then
            label="dec_${c}_${r}_${a}_n$n"
            calibrate "$c" "$r" "$a" "$node"
            # shellcheck disable=SC2046
            cell_run "$label" "$n" $((FRAMES * (${LOOPS[${c}_${r}_$a]} + 1))) $(dec_cmd "$c" "$r" "$a" "$node" "${LOOPS[${c}_${r}_$a]}")
        else
            label="enc_${c}_${r}_${a}_q${q}_n$n"
            # shellcheck disable=SC2046
            cell_run "$label" "$n" "$FRAMES" $(enc_cmd "$c" "$r" "$a" "$q" "$node" "$FRAMES")
        fi
    done <<<"$list"
    if ((BREAKDEV)) && [[ -n $igpu ]]; then
        # the iGPU decode must really work before its (absent) dGPU activity means anything: one untimed framemd5 pass against the CPU reference
        if [[ ${R11_DRY:-0} == 1 ]]; then echo "DRY igpu framemd5 check"; elif ffmpeg -hide_banner -nostdin -y -loglevel error -hwaccel vaapi -hwaccel_device "$igpu" -i "$(clip_of h264 1080p)" -pix_fmt yuv420p -f framemd5 "$R11_RUN/verify_igpu_h264_1080p.framemd5" 2>>"$R11_RUN/ffmpeg.err" \
            && md5_equal "$R11_RUN/verify_igpu_h264_1080p.framemd5" "$SCR/ref_h264_1080p.framemd5"; then echo PASS >"$R11_RUN/igpu-md5.txt"; else echo FAIL >"$R11_RUN/igpu-md5.txt"; fi
        # shellcheck disable=SC2046
        cell_run "break_decode_on_igpu" 1 "$FRAMES" $(dec_cmd h264 1080p hw "$igpu" 20)
    fi
    t0=$(mono_ns); [[ ${R11_DRY:-0} == 1 ]] || sleep "${R11_IDLE_S:-60}"; t1=$(mono_ns)
    echo "idle-after,0,0,0,0,$t0,$t1,0" >>"$R11_RUN/video.csv"
}
# analysis: one line per cell: fps (frames of the whole group over its wall time) and the activity check against the first idle window.
analyse() {
    local csv=$R11_RUN/video.csv smp=$R11_RUN/video.sampler/samples.csv idle bad=0 label n fr w0 w1 t0 t1 fail fps act
    [[ -s $smp ]] || { echo "no samples.csv: no activity check possible"; return 14; }
    idle=$(activity "$smp" "$(awk -F, '$1=="idle-before" {print $6}' "$csv")" "$(awk -F, '$1=="idle-before" {print $7}' "$csv")")
    echo "IDLE baseline (max vclk0 dclk0 vclk1 dclk1 mm_act rows): $idle"
    while IFS=, read -r label n fr w0 w1 t0 t1 fail; do
        [[ $label == label || $label == idle-* ]] && continue
        act=ok
        activity_ok "$smp" "$t0" "$t1" "$idle" || act=NO_VIDEO_ACTIVITY
        if [[ $label == break_decode_on_igpu ]]; then
            if [[ $fail != 0 ]]; then echo "BREAK decode_on_igpu: the iGPU decode itself failed (failed=$fail), so no activity says nothing -> PROBE INVALID"; bad=1
            elif [[ $(cat "$R11_RUN/igpu-md5.txt" 2>/dev/null) != PASS ]]; then echo "BREAK decode_on_igpu: the iGPU decode's framemd5 did not match the CPU reference (or was not checked) -> PROBE INVALID"; bad=1
            elif [[ $act == ok ]]; then echo "BREAK decode_on_igpu: activity check reported PASS -> PROBE INVALID"; bad=1
            else echo "BREAK decode_on_igpu: activity check reported FAIL -> ok (FAIL as required)"; fi
            continue
        fi
        fps=$(awk -v f="$((fr * n))" -v a="$w0" -v b="$w1" 'BEGIN { d = b - a; printf "%.1f", (d > 0 ? f / d : 0) }')
        echo "CELL $label procs=$n fps_total=$fps fps_per_proc=$(awk -v f="$fps" -v n="$n" 'BEGIN{printf "%.1f", f/n}') activity=$act failed=$fail"
        [[ $act == ok && $fail == 0 ]] || bad=1
    done <"$csv"
    return $bad
}

# ---- main -----------------------------------------------------------------------------------------------------------------------------
if ((PLAN)); then do_plan; exit 0; fi
if ((MAKE)); then do_make_inputs; exit $?; fi
if ((CPUREF)); then do_cpu_reference; exit $?; fi

if ((CPU)); then
    bad=0
    ok() { echo "ok   $*"; }
    nok() { echo "FAIL $*"; bad=$((bad + 1)); }
    T=$(mktemp -d "${R11_CPUCHECK_DIR:-$R11_REPO/runs}/r11-video-cpucheck.XXXXXX") || exit 2
    bash -n "$SELF" && ok "bash -n run_r11_video.sh" || nok "bash -n run_r11_video.sh"
    bash -n "$here/r11_common.sh" && ok "bash -n r11_common.sh" || nok "bash -n r11_common.sh"
    for c in flock nice timeout journalctl awk perl ffmpeg ffprobe; do command -v "$c" >/dev/null && ok "tool $c" || nok "tool $c missing"; done
    do_plan >"$T/plan.txt" && ok "plan prints ($(ncells) cells)" || nok "plan"
    ((SMOKE == 0)) && (($(ncells) > 60)) && ok "cell list size $(ncells)" || nok "cell list size"
    cells | sort | uniq -d | grep -q . && nok "duplicate cells" || ok "no duplicate cells"
    # ffmpeg capabilities, read from the encoder/decoder lists only (the device is not opened)
    enc=$(ffmpeg -hide_banner -encoders 2>/dev/null)
    for e in h264_vaapi hevc_vaapi av1_vaapi libx264 libx265 libsvtav1; do grep -qw "$e" <<<"$enc" && ok "ffmpeg encoder $e" || nok "ffmpeg encoder $e missing"; done
    ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -qx vaapi && ok "ffmpeg hwaccel vaapi" || nok "ffmpeg hwaccel vaapi missing"
    ffmpeg -hide_banner -filters 2>/dev/null | grep -qw psnr && ok "ffmpeg psnr filter" || nok "ffmpeg psnr filter missing"
    ffmpeg -hide_banner -h encoder=h264_vaapi 2>/dev/null | grep -q -- '-quality' && ok "h264_vaapi has -quality" || nok "h264_vaapi has no -quality (the quality cells would be rejected by ffmpeg)"
    # render nodes by device id: the two must differ, and the dGPU's must not be the iGPU's
    dn=$(render_for_id "$R11_DEVID") && ok "dGPU render node $dn" || nok "no render node with device id $R11_DEVID"
    ig=$(render_for_id 0x13c0) && ok "iGPU render node $ig (the deliberate break's target)" || echo "note: no iGPU render node (break 1 cannot run on this machine)"
    [[ -n ${dn:-} && ${dn:-} != "${ig:-}" ]] && ok "the two nodes differ" || nok "dGPU and iGPU nodes are not distinct"
    # dry-run command lines: the dGPU node is in every one, the iGPU node in none
    R11_DRY=1 inner_cmds=$(for k in "dec h264 1080p hw - 1" "enc av1 4k 40M default 1"; do read -r kind c r a q n <<<"$k"; if [[ $kind == dec ]]; then dec_cmd "$c" "$r" "$a" "${dn:-/dev/dri/renderD128}" 4; else enc_cmd "$c" "$r" "$a" "$q" "${dn:-/dev/dri/renderD128}" 300; fi; done)
    grep -q "vaapi_device ${dn:-x}" <<<"$inner_cmds" && grep -q "hwaccel_device ${dn:-x}" <<<"$inner_cmds" && ok "dry-run commands name the dGPU node" || nok "dry-run commands do not name the dGPU node"
    [[ -z ${ig:-} ]] || ! grep -q "$ig" <<<"$inner_cmds" && ok "no dry-run command names the iGPU node" || nok "a command names the iGPU node"
    # controls on a tiny CPU-made clip
    sz=160x120; nf=30
    ffmpeg -hide_banner -nostdin -y -loglevel error -f lavfi -i "testsrc2=size=$sz:rate=30,noise=alls=8:allf=t" -frames:v $nf -pix_fmt yuv420p -f rawvideo "$T/a.yuv" &&
        ffmpeg -hide_banner -nostdin -y -loglevel error -f lavfi -i "mandelbrot=size=$sz:rate=30" -frames:v $nf -pix_fmt yuv420p -f rawvideo "$T/b.yuv" && ok "tiny raw sources made" || nok "tiny raw sources"
    cpu_encode h264 "$T/a.yuv" $sz $nf "$T/a.mp4" 1M && cpu_framemd5 "$T/a.mp4" "$T/a.md5" && cpu_framemd5 "$T/a.mp4" "$T/a2.md5" && ok "tiny CPU clip + framemd5" || nok "tiny CPU clip"
    md5_equal "$T/a.md5" "$T/a2.md5" && ok "control: identical decodes compare equal" || nok "control: identical decodes compare unequal"
    # break 2: one corrupted byte in the stream
    cp "$T/a.mp4" "$T/a_bad.mp4"; sz_b=$(stat -c %s "$T/a_bad.mp4")
    printf '\xff' | dd of="$T/a_bad.mp4" bs=1 seek=$((sz_b / 2)) conv=notrunc 2>/dev/null
    cpu_framemd5 "$T/a_bad.mp4" "$T/bad.md5" 2>/dev/null
    md5_equal "$T/a.md5" "$T/bad.md5" && nok "BREAK corrupted byte: md5 comparison stayed equal (PROBE INVALID)" || ok "BREAK corrupted byte: md5 comparison FAILED as required"
    stream_ok "$T/a.mp4" $nf $sz && ok "control: stream_ok on the clip ($nf frames, $sz)" || nok "control: stream_ok"
    stream_ok "$T/a.mp4" $((nf + 1)) $sz && nok "break: wrong frame count accepted" || ok "BREAK wrong frame count: stream_ok FAILED as required"
    p=$(psnr_of "$T/a.mp4" "$T/a.yuv" $sz)
    psnr_ok "$p" 30 && ok "control: PSNR of clip against its own source $p dB >= 30 (tiny-clip floor)" || nok "control: PSNR $p"
    # break 3: PSNR against a different clip
    pb=$(psnr_of "$T/a.mp4" "$T/b.yuv" $sz)
    psnr_ok "$pb" 30 && nok "BREAK different source: PSNR $pb dB passed the floor (PROBE INVALID)" || ok "BREAK different source: PSNR ${pb:-n/a} dB FAILED the floor as required"
    psnr_ok "" 30 && nok "break: empty PSNR accepted" || ok "BREAK unreadable PSNR: FAILED as required"
    # activity check on synthetic samples (r11_sampler columns): idle vclk0 0; busy vclk0 800; iGPU-did-the-work window shows idle values
    {
        echo "# synthetic"; echo "mono_ns,read_us,uclk,gfxclk,sclk,act,thr,ithr,hot,edge,mem,pw,spw,fan,fanp,vclk0,dclk0,vclk1,dclk1,mm,pw,gts,ok"
        for t in 100 200 300 400 500 600; do echo "$t,5,456,100,100,0,0x0,0x0,40,40,40,10,10,0,0,0,0,0,0,0,16,320,1"; done
        for t in 1000 1100 1200; do echo "$t,5,456,100,100,0,0x0,0x0,40,40,40,40,40,0,0,800,700,0,0,35,16,320,1"; done
    } >"$T/samples.csv"
    idle=$(activity "$T/samples.csv" 100 600)
    activity_ok "$T/samples.csv" 1000 1200 "$idle" && ok "control: video activity in the busy window is seen (idle $idle)" || nok "control: busy window not seen"
    activity_ok "$T/samples.csv" 100 600 "$idle" && nok "BREAK idle window passed as video activity (PROBE INVALID)" || ok "BREAK iGPU-did-the-work window (idle values): activity check FAILED as required"
    activity_ok "$T/samples.csv" 5000 6000 "$idle" && nok "break: empty window accepted" || ok "BREAK window without samples: FAILED as required"
    # analyse() on a synthetic run: a good cell passes, the break cell reports FAIL as required
    mkdir -p "$T/run/video.sampler"
    cp "$T/samples.csv" "$T/run/video.sampler/samples.csv"
    {
        echo "label,n,frames_per_proc,wall_start,wall_end,mono_start_ns,mono_end_ns,failed"
        echo "idle-before,0,0,0,0,100,600,0"
        echo "dec_h264_1080p_hw_n1,1,3000,10.0,12.0,1000,1200,0"
        echo "break_decode_on_igpu,1,300,20.0,22.0,100,600,0"
    } >"$T/run/video.csv"
    echo PASS >"$T/run/igpu-md5.txt"
    (R11_RUN=$T/run; analyse >"$T/analysis.txt"; echo $? >"$T/analysis.rc")
    grep -q '^CELL dec_h264_1080p_hw_n1 procs=1 fps_total=1500.0 .*activity=ok' "$T/analysis.txt" && grep -q 'BREAK decode_on_igpu.*FAIL as required' "$T/analysis.txt" && [[ $(cat "$T/analysis.rc") == 0 ]] && ok "analyse(): good cell passes, iGPU break fails as required" || { nok "analyse() on the synthetic run"; cat "$T/analysis.txt"; }
    # deliberate breaks of the iGPU break itself: a failed iGPU decode, or one whose framemd5 was wrong or unchecked, must NOT count as a passed break
    sed 's/^break_decode_on_igpu,1,300,20.0,22.0,100,600,0$/break_decode_on_igpu,1,300,20.0,22.0,100,600,1/' "$T/run/video.csv" >"$T/run/video.failed.csv"
    mkdir -p "$T/runf" && cp -r "$T/run/video.sampler" "$T/runf/" && cp "$T/run/video.failed.csv" "$T/runf/video.csv" && echo PASS >"$T/runf/igpu-md5.txt"
    (R11_RUN=$T/runf; analyse >"$T/analysis-failed.txt"; echo $? >"$T/analysis-failed.rc")
    grep -q 'iGPU decode itself failed.*PROBE INVALID' "$T/analysis-failed.txt" && [[ $(cat "$T/analysis-failed.rc") != 0 ]] && ok "BREAK failed iGPU decode (exit 1, no dGPU activity): the break is NOT accepted" || nok "BREAK failed iGPU decode was accepted as a passed break (PROBE INVALID)"
    mkdir -p "$T/runm" && cp -r "$T/run/video.sampler" "$T/runm/" && cp "$T/run/video.csv" "$T/runm/video.csv" && echo FAIL >"$T/runm/igpu-md5.txt"
    (R11_RUN=$T/runm; analyse >"$T/analysis-md5.txt"; echo $? >"$T/analysis-md5.rc")
    grep -q 'framemd5 did not match' "$T/analysis-md5.txt" && [[ $(cat "$T/analysis-md5.rc") != 0 ]] && ok "BREAK iGPU framemd5 mismatch: the break is NOT accepted" || nok "BREAK iGPU framemd5 mismatch was accepted"
    rm -f "$T/runm/igpu-md5.txt"
    (R11_RUN=$T/runm; analyse >"$T/analysis-md5b.txt"; echo $? >"$T/analysis-md5b.rc")
    [[ $(cat "$T/analysis-md5b.rc") != 0 ]] && ok "BREAK iGPU framemd5 never checked: the break is NOT accepted" || nok "BREAK unchecked iGPU decode was accepted"
    [[ $PSNR_FLOOR == 26 || -n ${R11_PSNR_FLOOR:-} ]] && ok "default PSNR floor is 26 dB as in expect-4a.md" || nok "default PSNR floor $PSNR_FLOOR is not the 26 dB of expect-4a.md"
    grep -q 'default 26' "$R11_REPO/scripts/round11/expect/expect-4a.md" 2>/dev/null || grep -q 'floor = 26 dB' "$R11_REPO/scripts/round11/expect/expect-4a.md" && ok "expect-4a.md states the 26 dB floor" || nok "expect-4a.md does not state a 26 dB floor"
    # dry-run of the whole locked window with the card jobs replaced by echo (no ffmpeg call, no GPU)
    (R11_DRY=1 R11_IDLE_S=0 R11_RUN=$T/dry SMOKE=1 inner_run >"$T/dry.txt" 2>&1); grep -q '^DRY ' "$T/dry.txt" && ok "dry run prints card commands and runs none ($(grep -c '^DRY ' "$T/dry.txt") lines)" || nok "dry run"
    (R11_SELFTEST=1 r11_gate_selftest "$T/gate") >"$T/gate.txt" 2>&1; g=$(grep -c 'gate-selftest FAIL' "$T/gate.txt")
    ((g == 0)) && ok "gate self-test ($(grep -c 'gate-selftest PASS' "$T/gate.txt") cases pass, $(grep -c 'gate-selftest SKIP' "$T/gate.txt") skipped)" || { nok "gate self-test: $g case(s) failed"; grep FAIL "$T/gate.txt"; }
    r11_find_card && ok "card found by device id: $R11_PCI, render node $R11_RENDER" || nok "card lookup"
    grep -rn "$HOME" "$R11_REPO/src/round11" "$R11_REPO/scripts/round11" 2>/dev/null | grep -q . && nok "a path under \$HOME is written in src/round11 or scripts/round11" || ok "no \$HOME path in src/round11 or scripts/round11"
    rm -rf "$T"
    echo "cpu-checks: $bad problem(s)"
    exit $((bad ? 1 : 0))
fi

if ((RUN)); then
    [[ -n $INNER ]] && { inner_run; exit $?; }
    r11_find_card || exit 3
    R11_RUN=${OUT:-$R11_REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r11video}
    export R11_RUN
    mkdir -p "$R11_RUN"
    [[ -s $R11_REPO/scripts/round11/expect/expect-4a.md ]] && cp "$R11_REPO/scripts/round11/expect/expect-4a.md" "$R11_RUN/expect.md" || r11_die 2 "expect-4a.md missing: expectations come first"
    [[ -s $(clip_of h264 1080p) && -s $(raw_of 1080p) ]] || r11_die 2 "inputs missing in $SCR: run --make-inputs first (CPU, owner not gaming)"
    r11_log "run folder $R11_RUN"
    SMOKE_ARG=""; ((SMOKE)) && SMOKE_ARG=--smoke
    BRK_ARG=""; ((BREAKDEV)) && BRK_ARG=--break-device
    r11_locked_job video 1500 bash "$SELF" --run --inner go $SMOKE_ARG $BRK_ARG --out "$R11_RUN"
    rc=$?
    # the sampler dir of the job is named video.sampler by the wrapper
    ((rc == 0 || rc == 14)) && { analyse | tee "$R11_RUN/analysis.txt"; a=${PIPESTATUS[0]}; ((a != 0 && rc == 0)) && rc=14; }
    exit "$rc"
fi
echo "give a mode: --plan, --cpu-checks, --make-inputs, --cpu-reference or --run (see the header of $SELF)"
exit 2
