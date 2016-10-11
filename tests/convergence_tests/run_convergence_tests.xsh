#!/usr/bin/env xonsh

python run_convergence_tests.py > output

date = $(date --rfc-3339=date).strip()

mv output convergence_runs/@(date+'.stdout')
mv convergence_test_results convergence_runs/@(date+'.results')
mv tests convergence_runs/@(date+'.pickle')
