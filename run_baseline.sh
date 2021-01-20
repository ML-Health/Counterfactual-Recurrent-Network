#!/usr/bin/env bash

echo Training CRN...
# Trains CRN encoder and decoder on observational tumor growth dataset generated from rMSN baseline code
python test_crn.py --chemo_coeff=10 --radio_coeff=10 --results_dir=baseline_results --model_name=crn_baseline --data=obs --b_encoder_hyperparm_tuning=True --b_decoder_hyperparm_tuning=True

# Runs simulations using CRN trained above and saves %RMSEs (overall & over time) to results_dir
echo Simulating chemo only...
python test_crn.py --chemo_coeff=10 --radio_coeff=10 --results_dir=baseline_results --model_name=crn_baseline --data=chemo --simulate=True
echo Simulating radio only...
python test_crn.py --chemo_coeff=10 --radio_coeff=10 --results_dir=baseline_results --model_name=crn_baseline --data=radio --simulate=True
echo Simulating chemo+radio...
python test_crn.py --chemo_coeff=10 --radio_coeff=10 --results_dir=baseline_results --model_name=crn_baseline --data=chemrad --simulate=True
echo Simulating no treatment...
python test_crn.py --chemo_coeff=10 --radio_coeff=10 --results_dir=baseline_results --model_name=crn_baseline --data=notrt --simulate=True

echo Training and simulation finished! Simulation results saved to results_dir...



