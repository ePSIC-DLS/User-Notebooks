#!/bin/bash
#$ -l h_rt=48:00:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G
#$ -e /dls/science/groups/e02/Mohsen/code/jupyterhub_active/JEOL_EDX_Multiframe/cluster_submit/logs
#$ -o /dls/science/groups/e02/Mohsen/code/jupyterhub_active/JEOL_EDX_Multiframe/cluster_submit/logs


module load python/epsic3.7
conda activate sandbox
python /dls/science/groups/e02/Mohsen/code/jupyterhub_active/JEOL_EDX_Multiframe/cluster_submit/jeol_edx_process.py $1
