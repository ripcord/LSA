# The Disgruntled Cinema Manager Algorithm

## Description
An algorithm which uses tabu search to sort a random number of people who arrive to a theater over a finite period of time into a fixed number of theaters of varying limited capacities.

## Algorithm
We have a cinema with a fixed number of theaters. For our implementation (written in Python), we have:
  *5 Large (L = set of all large theaters)
  *3 Medium (M = set of all medium theaters)
  *2 Small (S = set of all small theaters)
  
We also have a fixed total theater capacity (total number of people which can occupy all theaters of a given theater set). This total capacity determines individual theater capacity, thus: c (individual theater capacity) = C (total theater capacity) / Ln (number of theaters in a given set of theaters, in this case set of large theaters).

In our implementation, moviegoers arrive to the cinema in groups (groups can also consist of a single individual). We randomly generate random groups 

## Contributors
sustac

Sannity

ripcord

Ixw123
