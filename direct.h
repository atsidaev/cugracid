#include "grid/Grid.h"
CUDA_FLOAT* CalculateDirectProblem(Grid& bottom, Grid& top, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double asimptota, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double dsigma, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, Grid* dsigma, int mpi_rank, int mpi_size);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size);
