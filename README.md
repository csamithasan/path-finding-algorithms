# Two Path Finding Algorithms
This repository presents the implementation and evaluation of two path-planning algorithms: Lifelong Planning A* (LPA*) and D* Lite in C++ programming language in an 8-connected grid world. Both algorithms were applied to different grid-based pathfinding problems to compare their performance under dynamic conditions. The aim is to evaluate their efficiency, adaptability, and scalability across different maps and heuristics. This program can execute algorithms using different parameters to record and analyze their performance. Heuristic functions: Euclidean distance and Chebyshev distance. A cross-platform graphics start-up program that is provided is used to plot the optimal path generated and view the details of the variables involved in the computations.

Implementation Overview
The system is implemented in C++ and integrates with a GridWorld environment. The algorithms were executed in two phases: initial planning and replanning after a environment change. Both Chebyshev and Euclidean heuristics were implemented and tested. The experimental setup automatically loads grid maps, runs the algorithms, and records performance metrics.
Key performance metrics collected for both algorithms include:
- Path length
- Number of expansions
- Maximum queue size
- Vertex accesses
- Runtime
