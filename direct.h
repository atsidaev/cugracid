#include "grid/Grid.h"
CUDA_FLOAT* CalculateDirectProblem(Grid& grid, double dsigma, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& grid, Grid& dsigma, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& grid, double asimptota, double dsigma, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& grid1, Grid& grid2, double dsigma, int mpi_rank, int mpi_size);
