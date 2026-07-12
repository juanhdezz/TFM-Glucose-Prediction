#!/bin/bash
#SBATCH --job-name=training_balanced_replaceBG        # Nombre del trabajo (cambiar para identificar correctamente)
#SBATCH --output=/home/juanhdez/results/replaceBG_job.out      # Archivo de salida
#SBATCH --error=/home/juanhdez/results/replaceBG_job.err      # Archivo de errores
#SBATCH --time=24:00:00             # Tiempo máximo de ejecución (ajusta según sea necesario)
#SBATCH --mem=32G                    # Memoria solicitada (Dejar fijo)
#SBATCH --cpus-per-task=6           # CPUs por tarea (Dejar fijo)
#SBATCH --gres=gpu:1                # Si usas GPU, esta línea es necesaria (elimina la linea si no la usas)

# Cargar el entorno de Conda (asegurarte de que esté bien configurado)
source /opt/miniconda/etc/profile.d/conda.sh # Esta línea es común para todos
conda activate glucose_balancing # Según el nombre del ambiente que hayas creado

# Ejecutar tu script Python dentro del entorno Conda (poner la ruta completa, no poner ruta relativa)
python /home/juanhdez/scripts/balancing_servidor.py --groups age --overwrite

# Desactivar el entorno de Conda (opcional)
conda deactivate