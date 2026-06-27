Guía básica para ejecutar scripts de Python con Slurm

Esta guía está pensada para usuarios nuevos que desean ejecutar scripts de Python en un servidor
que utiliza Slurm como gestor de trabajos. Slurm permite enviar tareas para que se ejecuten en
segundo plano, aprovechando los recursos del servidor de manera ordenada y eficiente.

1. ¿Qué es Slurm?

Slurm es un planificador de trabajos (job scheduler) que se usa en servidores o clústeres para
gestionar recursos computacionales. En lugar de ejecutar tu script directamente con python
script.py, lo colocas en una "cola de trabajos" y el sistema lo ejecuta cuando haya recursos
disponibles.

2. ¿Cómo se usa Slurm?

Para usar Slurm, se crea un script batch. Este script indica a Slurm cuántos recursos necesitas
(como memoria, CPU, GPU) y qué comandos ejecutar (por ejemplo, activar un entorno de
Conda y correr un script de Python).

3. Plantilla base de un script Slurm para Python

Aquí tienes un ejemplo que puedes usar como base:

(No incluyas esto como código, solo edítalo tú en un archivo nuevo con extensión .sh)

•  La parte superior (#SBATCH ...) le dice a Slurm qué recursos necesitas.
•  La parte inferior ejecuta tu código Python dentro de tu entorno Conda.

Tendrás que modificar lo siguiente:

•  --job-name: ponle un nombre representativo al trabajo.
•  --output y --error: son los archivos donde se guardará la salida normal y los errores.
•  --time: tiempo máximo que puede durar el trabajo (si se pasa, se cancela).
•  --mem: memoria RAM solicitada. Usar como máximo 32 GB en este caso.
•  --cpus-per-task: número de núcleos de CPU que usará. Usa 6 como máximo.
•  --gres=gpu:1: esta línea debe incluirse solo si tu script usa GPU, si no la usa no la

pongas. Esto es importante, tanto para que otros usuarios usen la gpu si no la necesitas tu,
como para que no te ponga el script en cola si no la necesitas esperando por el recurso.

•  conda activate: reemplaza tf15test con el nombre de tu entorno Conda.
•  python ...: reemplaza con la ruta completa a tu script Python.

4. Enviar el trabajo al servidor

Guarda el script como, por ejemplo, mi_trabajo.sh. Luego, desde la terminal, usa:

sbatch mi_trabajo.sh

Slurm te responderá con un mensaje como:
Submitted batch job 12345
Ese número es el ID de tu trabajo.

5. Ver el estado del trabajo

Usa estos comandos:

•  squeue: ver todos los trabajos en cola o ejecutándose
•  squeue -u TU_USUARIO: para ver los trabajos en cola o ejecutándose para tu usuario.
•  sacct -j ID: para ver información del trabajo una vez finalizado (reemplaza ID con el

número de tu trabajo).

•  scancel ID: para cancelar un trabajo en curso.

6. Ver resultados y errores

Cuando Slurm ejecute tu script, creará dos archivos:

•  Uno con el nombre que pusiste en --output, que tendrá todo lo que normalmente verías

en pantalla (como print()).

•  Otro con el nombre de --error, donde se almacenan errores (por ejemplo, si olvidaste

importar una librería o el entorno Conda no cargó bien).

7. Buenas prácticas

•  Siempre usa rutas absolutas para evitar errores.
•  Asegúrate de que tu entorno Conda esté correctamente configurado.
•  Si tu script necesita archivos de entrada, asegúrate de que estén en la ruta correcta.
•  No ejecutes trabajos pesados directamente en la terminal del servidor, usa Slurm.

7. Adjunto

Adjunto a esta guía se envía un ejemplo de archivo .sh de configuración nombrado
sbatch_script.sh por el cual puede servir de plantilla para crear el suyo.

