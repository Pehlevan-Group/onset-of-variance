import argparse

parser = argparse.ArgumentParser(description='Feed in depth, task, and sigma ')
parser.add_argument('-L', type=int, required=True, nargs='?', help="Depth")
parser.add_argument('-D', type=int, required=True, nargs='?', help="Input Dimension")
parser.add_argument('-N', type=int, required=False, nargs='?', default=1000, help="Width")
parser.add_argument('-P', type=int, required=False, nargs='?', help="Dataset size")
parser.add_argument('-t', type=str, required=True, nargs='+', help="Task")
parser.add_argument('-s', type=float, required=True, nargs='+', help="Initialization scale")
parser.add_argument('-n', type=int, required=False, default=5, nargs='?', help="Number of repeats")
parser.add_argument('-d', type=int, required=False, default=5, nargs='?', help="Number of datasets")
parser.add_argument('--subtract', type=int, required=False, default=1, help="Subtract f0")

args = parser.parse_args()

depth = args.L
dim = args.D
width = args.N
dataset_size = args.P
ts = args.t
sigmas = args.s
num_repeats = args.n
num_datasets = args.d
subtract = args.subtract


for t in ts:
  for sigma in sigmas:
    for d_key in range(num_datasets):
      if width is None:
        file_name = f"bash_scripts/L={depth}_D={dim}_t={t}_s={sigma:.2f}_d={d_key}.sh"
      else:
        file_name = f"bash_scripts/L={depth}_D={dim}_N={width}_t={t}_s={sigma:.2f}_d={d_key}.sh"
      with open (file_name, 'w') as rsh:
          rsh.write('''\
#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:1        # Number of GPUs (per node)
#SBATCH -p seas_gpu,gpu,gpu_requeue	    # Partition to submit to
#SBATCH --mem=100G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home00/aatanasov/Jupyter/out_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home00/aatanasov/Jupyter/out_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/2020.11
module load cuda/11.4.2-fasrc01
module load cudnn/8.2.2.26_cuda11.4-fasrc01

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
conda activate varlim
nvidia-smi
''')
          if t =="multimodal":
            rsh.write(f"/n/home00/aatanasov/.conda/envs/varlim/bin/python pn_multimodal.py -L {depth} -D {dim} -N {width} -s {sigma:.2f} -n {num_repeats} -d {d_key} \n")
          elif t=="cifar":
            rsh.write("/n/home00/aatanasov/.conda/envs/varlim/bin/python pn_sweeps_cifar.py -L {} -s {:.2f} -n {}\n".format(depth, sigma, num_repeats))
          elif t[:-1]=="train":
            rsh.write("/n/home00/aatanasov/.conda/envs/varlim/bin/python build_kernel.py -L {} -k {} -s {:.2f} -n {} -d {}\n".format(depth, t[-1], sigma, num_repeats, d_key))
          elif t[:-1]=="test":
            rsh.write("/n/home00/aatanasov/.conda/envs/varlim/bin/python build_kernel.py -L {} -k {} -s {:.2f} -n {} -d {} -t 1\n".format(depth, t[-1], sigma, num_repeats, d_key))
          elif t[:-1]=="tetr":
            rsh.write("/n/home00/aatanasov/.conda/envs/varlim/bin/python build_kernel.py -L {} -k {} -s {:.2f} -n {} -d {} --tetr 1 \n".format(depth, t[-1], sigma, num_repeats, d_key))
          elif width and dataset_size:
            rsh.write(f"/n/home00/aatanasov/.conda/envs/varlim/bin/python large_p.py -L {depth} -D {dim} -N {width} -P {dataset_size} -k {t} -s {sigma:.2f} -n {num_repeats} -d {d_key} \n")
          else:
            rsh.write(f"/n/home00/aatanasov/.conda/envs/varlim/bin/python pn_sweeps.py -N {width} -L {depth} -D {dim} -k {t} -s {sigma:.2f} -n {num_repeats} -d {d_key} --subtract {subtract} \n")