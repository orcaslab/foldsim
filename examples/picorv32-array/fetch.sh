#!/bin/bash
# fetch.sh: PicoRV32 (YosysHQ/picorv32, ISC) at the commit the gate was run on, into upstream/ (unmodified).
set -e
cd "$(dirname "$0")"; mkdir -p upstream
SHA=ef203c2b0a3fb793280f5114941416c425c5b461
curl -sSfo upstream/picorv32.v https://raw.githubusercontent.com/YosysHQ/picorv32/$SHA/picorv32.v
echo "YosysHQ/picorv32 $SHA (ISC)" > upstream/SOURCE
