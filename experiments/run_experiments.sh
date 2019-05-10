# Cautious MPC experiments
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_cautious_mpc_nperf=5.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_cautious_mpc_nperf=10.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_cautious_mpc_nperf=15.py

#SafeMPC with varying H
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=2_nperf=5_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=2_nperf=10_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=2_nperf=15_r=1_beta_safe=2.py

# SafeMPC experiments with nperf=15
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=1_nperf=15_r=1_beta_safe=2.py
#python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=2_nperf=15_r=1_beta_safe=2.py # This is already done in the previous set of experiments
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=3_nperf=15_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=4_nperf=15_r=1_beta_safe=2.py

# SafeMPC experiments with nperf=0
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=1_nperf=0_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=2_nperf=0_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=3_nperf=0_r=1_beta_safe=2.py
python run.py --scenario_config journal_experiment_configs/cart_pole_rl_nsafe=4_nperf=0_r=1_beta_safe=2.py
