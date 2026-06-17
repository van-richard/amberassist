#!/bin/bash
# Prepare QMMM free energy simulations 
# Umbrella sampling 

set -euo pipefail

QMINFO="qm_info.txt"
[[ -f "$QMINFO" ]] || { echo "ERROR: missing $QMINFO" >&2; exit 1; }

# -------------------------
# Read qm_info.txt (SAFE)
# -------------------------
qmmask=""
qmtheory=""
qmhub_mode=""
qmcharge=""
thermostat=""
NSTEPS5=""
NSTEPS6=""
n_windows=""
print_freq=""

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  val="${line#*=}"

  # trim whitespace around key
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"

  case "$key" in
    qmmask)      qmmask="$val" ;;
    qmtheory)    qmtheory="$val" ;;
    qmhub_mode)  qmhub_mode="$val" ;;
    qmcharge)    qmcharge="$val" ;;
    thermostat)  thermostat="$val" ;;
    nsteps5)     NSTEPS5="$val" ;;
    nsteps6)     NSTEPS6="$val" ;;
    n_windows)   n_windows="$val" ;;
    print_freq)  print_freq="$val" ;;
  esac
done < "$QMINFO"

# sanity checks
for v in qmmask qmtheory qmcharge thermostat NSTEPS5 NSTEPS6 n_windows print_freq; do
  [[ -n "${!v}" ]] || { echo "ERROR: $QMINFO missing '$v='" >&2; exit 1; }
done

if [[ "${qmtheory}" == "EXTERN" ]]; then
  qmhub_mode="${qmhub_mode:-DFT}"
  case "${qmhub_mode}" in
    DFT|MTS) ;;
    *)
      echo "ERROR: qmhub_mode must be 'DFT' or 'MTS' when qmtheory=EXTERN (got '${qmhub_mode}')" >&2
      exit 1
      ;;
  esac
fi

# --- Thermostat block (drop this somewhere after THERMOSTAT is set) ---
case "${thermostat}" in
  langevin)
    THERMO_BLOCK=$'    ntt=3,         ! Langevin dynamics
    gamma_ln=1.0,  ! Friction coefficient (ps^-1)
    temp0=300.0,   ! Target temperature'
    ;;
  sinr)
    THERMO_BLOCK=$'    ntt=12,         ! SINR thermostat
    gamma_ln=1.0,  ! Friction coefficient (ps^-1)
    tempi=10.0,    ! Initial temp -- give it some small random velocities
    temp0=300.0,   ! Target temperature
    nkija=4,
    sinrtau=0.05,'
    ;;
  *)
    echo "ERROR: THERMOSTAT must be 'langevin' or 'sinr' (got '${thermostat}')" >&2
    exit 1
    ;;
esac

# --- QMCONFIG + QMHUB blocks (after QMTHEORY is set) ---
if [[ "${qmtheory}" == "EXTERN" ]]; then
  QMHUB_BLOCK=$'  &qmhub
   config="qmhub.ini",
   basedir="__QMHUBSCRATCH__",
  /\n'

  QMCONFIG_BLOCK=$'    ! Shake
    qmshake=0,     ! Use Shake for QM atoms

    ! Potential energy control
    qmcut=999.0,    ! Cutoff for QM/MM electrostatic interactions
    qm_ewald=0,    ! QM/MM with periodic boundaries
    qm_pme=0,      ! Use PME for QM-MM electrostatic interactions
    qmmm_switch=0, ! Switching for QM-MM
    writepdb=1,    ! Check QM atoms
    /\n'
else
  QMHUB_BLOCK=""  # delete __QMHUB__ line

  QMCONFIG_BLOCK=$'    ! Shake
    qmshake=1,     ! Use Shake for QM atoms

    ! Potential energy control
    qmcut=10.0,    ! Cutoff for QM/MM electrostatic interactions
    qm_ewald=1,    ! QM/MM with periodic boundaries
    qm_pme=1,      ! Use PME for QM-MM electrostatic interactions
    qmmm_switch=1, ! Switching for QM-MM
    writepdb=1,    ! Check QM atoms
    /\n'
fi

render_mdin() {
  local tmpl="$1"
  local fname="$2"   # must be "step5" or "step6"
  local out="${fname}.00_equilibration.mdin"

  local nsteps
    case "${fname}" in
    step5) nsteps="${NSTEPS5}" ;;
    step6) nsteps="${NSTEPS6}" ;;
    *)
      echo "ERROR: render_mdin fname must be 'step5' or 'step6' (got '${fname}')" >&2
      exit 1
      ;;
  esac

  awk -v thermo="$THERMO_BLOCK" \
      -v qmhub="$QMHUB_BLOCK" \
      -v qmcfg="$QMCONFIG_BLOCK" \
      -v qmmask="$qmmask" \
      -v qmtheory="$qmtheory" \
      -v qmcharge="$qmcharge" \
      -v nsteps="$nsteps" \
      -v print_freq="$print_freq" \
      -v fname="$fname" '
  function repl(s, token, val,   pos, tlen) {
    tlen = length(token)
    while ((pos = index(s, token)) > 0) {
      s = substr(s, 1, pos-1) val substr(s, pos+tlen)
    }
    return s
  }

  index($0,"__THERMOSTAT__") { printf "%s", thermo; next }
  index($0,"__QMHUB__")      { if (length(qmhub)>0) printf "%s", qmhub; next }
  index($0,"__QMCONFIG__")   { printf "%s", qmcfg; next }

  {
    $0 = repl($0, "__QMMASK__",    qmmask)
    $0 = repl($0, "__QMTHEORY__",  qmtheory)
    $0 = repl($0, "__QMCHARGE__",  qmcharge)
    $0 = repl($0, "__NSTEPS__",    nsteps)
    $0 = repl($0, "__FNAME__",     fname)
    $0 = repl($0, "__PRINTFREQ__", print_freq)
    print
  }
' "$tmpl" > "$out"
}


# write file:
render_mdin "equilibration.mdin.tmp" "step5"
render_mdin "equilibration.mdin.tmp" "step6"

n_win=$((n_windows - 1))
seq -w 0 "${n_win}" > ../list
