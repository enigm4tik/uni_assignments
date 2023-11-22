# Conway's Game of Life
 
### Introduction
Introduced in 1970 by John Conway  
Cellular automaton  
Zero-player game  
Game Board: Two dimensional orthogonal grid, every field is a cell  
Cells can either be alive or dead  

### Rules
Every cell has 8 neighbours  

Rule 1: Any dead cell with exactly three living neighbours becomes alive 
Rule 2: Any live cell with two or three living neighbours stays alive  
Rule 3: Any live cell with fewer than two living neighbours dies  
Rule 4: Any live cell with more than three living neighbours dies  

### Exercise 1
Program Convey's Game of Life

* Read start pattern from text file 
    * fields are determined in first line - eg. 100, 100
    * x - live cell
    * . - dead cell
* Compute in two different ways
    * Single threaded - sequential
    * Multi threaded - OpenMP
* Measure setup, calculation and finalization time for each method 
* Implement Line Parameters
    * --load <filename>
    * --save <filename>
    * --generations <n> (run n generations)
    * --measure (print time measurements)
    * --mode seq (sequential version)
    * --mode omp --threads <num> (parallel OpenMP version with NUM threads)
