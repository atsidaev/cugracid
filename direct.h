#include "grid/Grid.h"
FLOAT* CalculateDirectProblem(Grid& grid, double dsigma, int mpi_rank, int mpi_size);
FLOAT* CalculateDirectProblem(Grid& grid, double asimptota, double dsigma, int mpi_rank, int mpi_size);
FLOAT* CalculateDirectProblem(Grid& grid1, Grid& grid2, double dsigma, int mpi_rank, int mpi_size);