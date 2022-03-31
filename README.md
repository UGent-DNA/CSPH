# Capacitated Shortest Path Heuristic
This program presents CSPH, a constructive heuristic for the Capacitated Steiner Tree problem. The algorithm is built for sparse graphs and runs in practice close to O(|V|^2) for these graphs.


### CSPH algorithm
The main algorithm can be found in sample/algorithm/csph.py. It also contains the experiments for the paper. These write results to output/results.txt. 

### Result visualisation
Then you can create different visualisations from sample/interpret_results/read_results.py based on the output/results_all.txt file (you can copy paste from output/results.txt or change names).

### DATA
The input is graph data in the ".stp" format that is described at http://steinlib.zib.de/format.php
The data is transformed from file to custom graph class (sample/datastructures/) in the sample/general/read_data.py file.
