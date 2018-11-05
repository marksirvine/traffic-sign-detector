#!/bin/bash                                                                                                                                          
#SBATCH -t 0-02:00 # Runtime in D-HH:MM                                                                                                              
#SBATCH -p gpu # Partition to submit to                                                                                                              
#SBATCH --gres=gpu:1                                                                                                                                 
#SBATCH --mem=20000
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J  tsr   # name                                                                                                                                  
#SBATCH -o tsr_%j.out # File to which STDOUT will be written                                                                                    
#SBATCH -e tsr_%j.err # File to which STDERR will be written                                                                                    
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL                                                                             
#SBATCH --mail-user=<ab12345>@bristol.ac.uk # Email to which notifications will be sent                                                                

module add libs/tensorflow/1.2
srun python tsr.py
wait
