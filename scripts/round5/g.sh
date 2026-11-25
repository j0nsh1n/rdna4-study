#!/usr/bin/env bash
# usage: g.sh <jobname> <hot-start C> <timeout s> <cmd...>   one gated, watched GPU job (profile Default)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
cd "$REPO" || exit 2
name=$1; hot=$2; tmo=$3; shift 3
export LLM_EXPECT_PROFILE=Default LLM_RUN=$R
S=scripts/study-llm.sh
$S dev "$name" "$hot" "$tmo" "$S" __watch "$R/watch/$name" "$name" "$@"
rc=$?
echo "$name rc=$rc"
exit $rc
