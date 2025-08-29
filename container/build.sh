#! /bin/bash

# YL: taken from https://github.com/legacysurvey/legacypipe/blob/main/docker-nersc/build.sh
# commit d5d0e48a2deb07a2fd04b9d2445b0906daea8ea1

# Tunnel to NERSC's license server
ssh -fN intel-license-nersc

if [ "$(uname)" = "Darwin" ]; then
    # Mac OSX
    LOCAL_IP=$(ipconfig getifaddr $(route get nersc.gov | grep 'interface:' | awk '{print $NF}'))
else
    # Linux
    LOCAL_IP=$(ip route ls | tail -n 1 | awk '{print $NF}')
fi

docker build --add-host "intel.licenses.nersc.gov:${LOCAL_IP}" -t $1 .