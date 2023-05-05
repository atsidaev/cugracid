#include "../grid/Grid.h"
#include <vector>
CUDA_FLOAT* CalculateDirectProblem(Grid& bottom, Grid& top, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
CUDA_FLOAT* CalculateDirectProblem(Grid& bottom, Grid& top, double dsigma, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double asimptota, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double dsigma, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, Grid* dsigma, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
CUDA_FLOAT* CalculateDirectProblem(Grid& g, double dsigma, Grid* dsigmaGrid, int mpi_rank, int mpi_size, std::vector<unsigned char> devices_list);
