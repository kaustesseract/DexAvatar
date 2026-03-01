#!/bin/bash

bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M3.5_hamer_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose.sh"
wait
echo "All finished"
