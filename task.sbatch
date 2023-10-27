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
srun -n 2 python3 main.py --use_agent_graph=0 --total_timesteps=6000 --run_name=baseline
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.0013 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=3000 --graph_param2=2 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=5 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=3 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=7 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=15 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=20 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ws --graph_param1=40 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=5 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.0005 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.001 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.0013 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.0019 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=er --graph_param1=0.0023 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=5 --graph_param2=0.1 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=1500 --graph_param2=2 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=2500 --graph_param2=2 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=3000 --graph_param2=2 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=3500 --graph_param2=2 --run_name=late_intervention
srun -n 2 python3 main.py --graph_gen=ba --graph_param1=6000 --graph_param2=2 --run_name=late_intervention


done