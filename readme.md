Pierre ZACHARY
2183251

# Rendu OpenCL

## Prérequis
- cmake 
- opencl ( testé avec la version 3.0 de nvidia )
- éventuellement openmp pour la comparaison cpu 

## Compilation
```
mkdir build
cd build
cmake ..
make
```

## Utilisation
```
./opencl
```

## Résultats pour la matrice 1025x1025
```
CPU : 231948600 ns
CPU avec openmp : 43047500 ns
GPU sans chunks : 7870500 ns
GPU avec chunks de 128x128 : 3529500 ns
GPU avec chunks de 1024x1024 : 1325900 ns
``` 

## Specs de la machine utilisée pour les tests
```
CPU : Intel(R) Core(TM) i7-8700K CPU @ 5.00GHz 6/12 
GPU : GeForce RTX 3060 Ti 8GB
    Device Max Compute Units: 38
    Device Global Memory: 8589279232
    Device Max Clock Frequency: 1710
    Device Max Allocateable Memory: 2147319808
    Device Local Memory: 49152
    Max Work-group Total Size: 1024
```

## Explication de l'implementation 
### CPU
Pour la version CPU j'ai repris la proposition d'algorithme par itération sur la mnt : 
- Pour chaque cellule de la mnt on commence par calculer la direction de celui-ci vers le voisin le plus bas ou lui-même 
- On effectue ensuite un nombre indeterminé d'itérations sur la mnt : 
  - pour chaque cellule on regarde si la cellule a des dépendences ( voisin qui pointe vers elle ), si c'est le cas, on s'assure que ces dépendances ont fini leurs propres calculs
  - si elles ont fini leurs propres calculs on calcule la somme de toutes ces cellules et on ajoute 1 pour obtenir la valeure de la cellule courante 
  - si il n'y a aucune dépendance la cellule courante vaut 1
- On continu ainsi jusqu'à ce que toutes les cellules aient été traitées

### GPU
Pour la version GPU j'avais commencé par faire la même chose que pour la version CPU, mais j'ai remarqué que le temps d'exécution était très long dû au fait qu'il faille placer une barrière entre chaque itération, j'ai donc cherché à optimiser le code pour me débarrasser de cette barrière.
Ma solution est donc la suivante : 
- Chaque worker va s'occuper de traiter une cellule i,j de la mnt correspondant à son id 
- Le worker en question commence par calculer la direction de sa propre cellule
- Puis ce worker appel une fonction récursive qui va : 
  - ajouter 1 à la cellule courante via un atomicAdd ( permet de s'assurer que les writes sur cette cellule ne sont pas fait en conccurence )
  - regarder si la cellule courante a une direction ( flowDir ne vaut pas 0 ), on doit la calculer si ce n'est pas le cas ( un même worker peut être ammené à calculer plusieurs cases de flowDir, cependant ces cases sont toutes stocké en mémoire globale et donc elles ne seront de toute façon calculer qu'une fois )
  - si c'est le cas on répète la fonction sur la cellule pointée par la direction de la cellule courante
- Cela permet d'obtenir le même résultat que la version CPU mais sans avoir besoin de barrière entre chaque itération

### Optimisation envisagée : 
- J'ai commencé par run le kernel en 2 dimensions, cependant cela rajoutait en grand nombre de run à effectuer pour la version par chunks dans le cas où la taille de la matrice n'est pas multiple ( voir la suite )
- L'avantage des atomic add par rapport aux barrier est que la carte s'occupe toute seule de scheduler les add sans que les worker n'aient besoin d'attendre alors que la barrier force les workers à s'attendre
- Il n'est pas possible de charger la mnt en mémoire locale avec ma version car on ne sait pas à l'avance quelles cases vont être nécessaire ( étant donné qu'un même worker peut être amené à calculer plusieurs cases ), de plus il y a de forte chance que cette matrice dépasse la taille autorisé pour worker group en mémoire locale ( taille max est de 49152 pour ma carte, la taille de la matrice complète serait de 4*1025*1025 = 41943040 ce qui dépasse largement )

### Version par chunks
- Pour répondre à la question des grandes mnt, on peut envisager de run le kernel on plusieurs fois. Pour cela une première approche a été de diviser la matrice en plusieurs blocs 2D, cependant cela n'est pas utile car j'ai écrit mon kernel de tel sorte qu'il n'y ai pas de dépendance entre worker ( en dehors de l'atomic add ), et donc il est plus efficace de faire des chunks 1D, pour réduire de nombre de calculs superflux.
- L'avantage de cette approche, c'est que lorsque l'on schedule plusieurs kernels à la suite, opencl se charge de les runs de la manière le plus optimisé possible, ( pas forcément dans l'ordre, mais cela n'a pas d'importance ). On peut donc optimisé à fond le nombre de work groups et de workers dans un même groupe, ce qui explique les gains de performance entre la version classique 1025x1025 et la version par chunks.  

### Réponse aux questions : 
En cas de très grands MNT est-ce que le calcul sur GPU reste pertinent ? Comment peut-on gérer ce gros volume de données ?
- La contrainte de ma solution est qu'il y a forcément besoin de charger 1. La mnt 2. La matrice de flowDir 3. La matrice de résultat en mémoire globale, ce qui correspond à : 3 * 4 * 1025 * 1025 octets. On peut éventuellement imaginer libérer l'espace mémoire de la mnt une fois que la matrice de direction est chargée, et on peut imaginer changer les types des deux autres matrices par : uchar pour la matrice flowDir, uint16 pour la matrice flowAcc ( probablement suffisant ), ce qui réduirait le nombre d'octets nécessaire au calcul à : 3 * 1025 * 1025, plus de 4 fois moins qu'actuellement. 
- Cependant, il ne me semble pas possible d'effectuer cet algorithme en l'état sans avoir un accès à toute la matrice de flow direction et toute la matrice de flow d'accumulation, où du moins toutes les cases pointés nécessaires en partant de la case initiale, ce qui peut au pire des cas correspondre à l'entièreté de la matrice.
- En revanche, du fait de ma solution par chunks, il n'y a pas de limites au nombre de workers nécessaire pour effectuer le calcul, ces derniers étant tous indépendants les uns des autres.

En considérant que les MNT peuvent contenir des cellules avec des NO_DATA (absence de données) pour lesquelles aucun calcul n'est effectué et qui ne sont pas prises en compte également dans un voisinage
- Cette problématique est déjà prise en compte dans mon code, les cases avec des NO_DATA ne sont pas prises en compte dans le calcul de la direction, et donc ne sont pas prises en compte dans le calcul de l'accumulation de flow.

Est-il possible d'en tenir compte pour votre version GPU ? Est-ce que ç'apporte un gain ?
- Par rapport aux nodata : même s'ils sont pris en compte il y aura forcément des workers qui n'auront finalement rien à faire, car ils seront sur l'une de ces cases, cependant à moins de faire un calcul en avance sur le CPU pour donner un offset aux workers sur le GPU et leur indiquer ce qu'ils doivent calculer pour éviter les nodata, je ne vois pas comment mieux optimiser cela. 
- Par ailleurs, la version par chunks étant executé "automatiquement en parallèle" ( selon les disponibilités du GPU ), il n'y aura pas réellement de perte de puissance de calcul car les workers ayant fini de travailler en avance ( nodata ), seront directement réattribués à la suite du calcul ( ils n'ont pas besoin d'attendre que les autres aient fini ). 

Pour une architecture à mémoire distribuée, comment paralléliseriez vous les différents calculs en tenant compte des NO_DATA ? Est ce qu'il y a un intérêt à utiliser du RMA ou un programme leader/followers ?
- Effectivement avec une version à mémoire distribué on pourrait utiliser une version Leader/Follower pour éviter les nodata. De cette façon, si on donne une ligne de nodata à un follower et que celui-ci termine en avance, on peut immédiatement lui donner une nouvelle ligne sans attendre que les autres aient fini.
- On pourrait aussi envisager une version RMA ou via ISend / IRecv, avec chaque worker qui possède sa matrice de flowAcc locale et on fait une réduction de ces matrices à la fin en faisant la somme de chaque case. 
- Il faudrait effectuer des benchmarks pour voir si le temps d'effectuer les communications avec chaque follower reste inférieur au temps de calcul 