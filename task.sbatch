#!/bin/bash

#!/bin/bash
#
#SBATCH --job-name=agentsim # Job name for tracking
#SBATCH --partition=cpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=20     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=30000            # 60GB RAM
#SBATCH --exclusive=mcs        # Exclusive mode, only this job will run
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job
#SBATCH --nodes=1
#SBATCH --ntasks=2

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh


## Execute your program(s) ##
for i in {0..4}; do
echo "iteration $i"
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=er --graph_param1=0.00133377792 --total_timesteps=6000 --run_name=erdos_renyi_low_baseline
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=er --graph_param1=0.0053351117 --total_timesteps=6000 --run_name=erdos_renyi_high_baseline
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=ws --graph_param1=5 --graph_param2=0.1 --total_timesteps=6000 --run_name=watts_strogatz_baseline
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=ws --graph_param1=17 --graph_param2=0.1 --total_timesteps=6000 --run_name=watts_strogatz_baseline
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=ba --graph_param1=2 --total_timesteps=6000 --run_name=barabasi_albert_low_baseline
srun -n 2 python3 main.py --intervention=0 --use_agent_graph=1 --graph_gen=ba --graph_param1=8 --total_timesteps=6000 --run_name=barabasi_albert_high_baseline
done