#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH --nodes=1 --ntasks=28 --gpus-per-node=1
#SBATCH --job-name=vocoder_TTS_project_time_rev_pfp_loss_dev
# account for CSE 5441 Au'21
#SBATCH --account=PAS2400
export PGM=vocoder_train.py    # <--- CHANGE THIS
export SLURM_SUBMIT_DIR=/fs/scratch/PAS2400/TTS_baseline_cpy/Real-Time-Voice-Cloning # <--- CHANGE THIS, everything else stay the same
export ARGUMENT=training_data/

echo job started at `date`
echo on compute node `cat $PBS_NODEFILE`

cd /fs/scratch/PAS2400/TTS_baseline_cpy
module load miniconda3
source activate tts

cd ${SLURM_SUBMIT_DIR} # change into directory with program to test on node
# /users/PAS1211/osu1053/CSE_5441/transform_library_mtx
# cp PCS_data_t00100 $TMPDIR #TMP is per node local storage on the cluster for good performance from data files
cd $TMPDIR # be in directory that you are reading data from

# echo job started at `date` >>current.out
# time ${SLURM_SUBMIT_DIR}/${PGM} < PCS_data_t00100  >> current.out 2>&1 # redirect stderr and stdout to the output file

CUDA_VISIBLE_DEVICES='0'
python ${SLURM_SUBMIT_DIR}/${PGM} run_id2 /fs/scratch/PAS2400/TTS_baseline/${ARGUMENT}  # <--- CHANGE THIS for arguments

# export SAVEDIR=${SLURM_SUBMIT_DIR}/tests/data_test.${SLURM_JOBID}
# mkdir ${SAVEDIR}
# mv current.out ${SAVEDIR}
