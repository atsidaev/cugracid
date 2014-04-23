#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "global.h"
#include "cuda/Vz.h"
#include "cuda/info.h"

#include "grid/Grid.h"

FLOAT* CalculateDirectProblem(Grid& g, Grid& top, double dsigma, int mpi_rank, int mpi_size)
{
	int mpi_rows_portion = g.nRow / mpi_size;
	printf("Every MPI thread will process %d rows\n", mpi_rows_portion);

	int grid_length = g.nCol * g.nRow;

	FLOAT *result;
	if (mpi_rank == MPI_MASTER)
	{
		result = new FLOAT[mpi_size * grid_length];
		memset(result, 0, mpi_size * grid_length * dsize);
	}
	else
	{
		result = new FLOAT[grid_length];
		memset(result, 0, grid_length * dsize);
	}
	

	if (!CalculateVz(top.data, g.data, result, g.nCol, g.nRow, mpi_rows_portion * mpi_rank, mpi_rows_portion))
	{
#ifdef USE_MPI		
		MPI_Abort(MPI_COMM_WORLD, 0);
#endif
		return NULL;
	}

	if (mpi_rank != MPI_MASTER)
	{
#ifdef USE_MPI		
		MPI_Send(result, grid_length, MPI_DOUBLE, MPI_MASTER, 0, MPI_COMM_WORLD);
#endif
	}
	else
	{
#ifdef USE_MPI		
		for (int i = 1; i < mpi_size; i++)
		{
			MPI_Recv(&result[i * grid_length], grid_length, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
#endif		
		for (int i = 1; i < mpi_size; i++)
		{
			for (int j = 0; j < grid_length; j++)
				result[j] += result[i * grid_length + j];
		}

		for (int j = 0; j < grid_length; j++)
			result[j] *= GRAVITY_CONST * dsigma;
		
		printf("%f\n", result[(g.nRow / 2) * g.nCol + g.nCol / 2]);
	
		return result;
	}
	
	return NULL;

}

FLOAT* CalculateDirectProblem(Grid& g, double dsigma, int mpi_rank, int mpi_size)
{
	int grid_length = g.nCol * g.nRow;

	Grid g2 = Grid::GenerateEmptyGrid(g);
	FLOAT *top = new FLOAT[grid_length];
	double assimptota = g.get_Average();
	for (int i = 0; i < grid_length; i++)
		top[i] = assimptota;

	g2.data = top;

	return CalculateDirectProblem(g, g2, dsigma, mpi_rank, mpi_size);
}
