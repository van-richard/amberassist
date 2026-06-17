#!/bin/bash
# Render a single classical MD+PLUMED mdin file into the parent run directory.

set -euo pipefail

MDINFO="md_info.txt"
TMPL="step5.00_equilibration.mdin"
OUT="../step5.00_equilibration.mdin"
[[ -f "$MDINFO" ]] || { echo "ERROR: missing $MDINFO" >&2; exit 1; }
[[ -f "$TMPL" ]] || { echo "ERROR: missing template $TMPL" >&2; exit 1; }

thermostat=""
NSTEPS=""
print_freq=""

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  val="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"

  case "$key" in
    thermostat) thermostat="$val" ;;
    nsteps) NSTEPS="$val" ;;
    print_freq) print_freq="$val" ;;
  esac
done < "$MDINFO"

for v in thermostat NSTEPS print_freq; do
  [[ -n "${!v}" ]] || { echo "ERROR: $MDINFO missing '$v='" >&2; exit 1; }
done

case "$thermostat" in
  langevin)
    THERMO_BLOCK=$'    ntt=3,         ! Langevin dynamics
    gamma_ln=1.0,  ! Friction coefficient (ps^-1)
    temp0=300.0,   ! Target temperature'
    ;;
  sinr)
    THERMO_BLOCK=$'    ntt=12,        ! SINR thermostat
    gamma_ln=1.0,  ! Friction coefficient (ps^-1)
    tempi=10.0,    ! Initial temp -- give it some small random velocities
    temp0=300.0,   ! Target temperature
    nkija=4,
    sinrtau=0.05,'
    ;;
  *)
    echo "ERROR: THERMOSTAT must be 'langevin' or 'sinr' (got '$thermostat')" >&2
    exit 1
    ;;
esac

awk -v thermo="$THERMO_BLOCK" \
    -v nsteps="$NSTEPS" \
    -v print_freq="$print_freq" '
function repl(s, token, val,   pos, tlen) {
  tlen = length(token)
  while ((pos = index(s, token)) > 0) {
    s = substr(s, 1, pos-1) val substr(s, pos+tlen)
  }
  return s
}

index($0,"__THERMOSTAT__") { printf "%s", thermo; next }

{
  $0 = repl($0, "__NSTEPS__", nsteps)
  $0 = repl($0, "__PRINTFREQ__", print_freq)
  print
}
' "$TMPL" > "$OUT"

[[ -s "$OUT" ]] || { echo "ERROR: output is empty: $OUT" >&2; exit 1; }
