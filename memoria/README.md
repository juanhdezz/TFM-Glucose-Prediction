# Instrucciones

## Datos personales

Primero cambia el fichero ```datos.tex``` y rellena tus datos personales, los de tu tutor, y los del TFM. Verás que hay algunos campos que pueden estar vacíos, para dejarlos vacíos simplemente borra el texto dentro de las llaves, por ejemplo, para eliminar el co-tutor hay que dejar vacío el campo ```\tutorB``` de esta forma:

```latex
\newcommand{\tutorB}{}
```

## Resúmenes y agradecimientos

Después puedes cambiar los ficheros de resúmenes:

- ```resumen.tex```: Aquí se encuentra el resumen en español.
- ```resumen-en.tex```: Aquí se encuentra el resumen en inglés.
- ```agradecimientos.tex```: Aquí se encuentran los agradecimentos a tutores, familia o amigos.

## Escribir tu TFM

Para escribir el TFM tan solo tendrás que modificar, en principio, los ficheros de la carpeta ```capitulos```. Fíjate que puedes añadir todas las imágenes en la carpeta ```imagenes```, y todas las tablas en la carpeta ```tablas```.

Fíjate que no tienes que añadir las extensiones ni el directorio donde se encuentran en las imágenes. De esta forma, puedes referenciar una imagen llamada ```imagenes/logo.png``` simplemente como ```logo```. La plantilla ya se encarga de buscar la mejor imagen (por orden: PDF, PNG o JPG).

## Cambiar la estructura y paquetes

La estructura se encuentra en el fichero ```tfm.tex```. A menos que necesites añadir paquetes especiales de \LaTeX, o requieras añadir apéndices o glosarios, no tendrías por qué tocarlo en principio.

También es útil cambiar el fichero si quieres crear órdenes personalizadas con ```\newcommand``` o ```\newenvironment```. Revisa la documentación de overleaf al respecto [Environments](https://www.overleaf.com/learn/latex/Environments).

En cualquier caso, fíjate en las guías y comentarios que aparecen dentro del fichero.
