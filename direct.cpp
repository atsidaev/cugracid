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

FLOAT* CalculateDirectProblem(Grid& bottom, Grid& top, double dsigma, int mpi_rank, int mpi_size)
{
	int mpi_rows_portion = bottom.nRow / mpi_size;
	if (bottom.nRow % mpi_size != 0)
	    mpi_rows_portion++;

	if (mpi_size > 1)
		printf("Every MPI thread will process %d rows\n", mpi_rows_portion);

	int grid_length = bottom.nCol * bottom.nRow;

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
	

	if (!CalculateVz(top.data, bottom.data, result, bottom.nCol, bottom.nRow, mpi_rows_portion * mpi_rank, mpi_rows_portion, bottom.xLL, bottom.yLL, bottom.xSize, bottom.ySize))
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
		
		// printf("%f\n", result[(bottom.nRow / 2) * bottom.nCol + bottom.nCol / 2]);
	
		return result;
	}
	
	return NULL;

}

FLOAT* CalculateDirectProblem(Grid& g, double asimptota, double dsigma, int mpi_rank, int mpi_size)
{
	int grid_length = g.nCol * g.nRow;

	Grid g2 = Grid::GenerateEmptyGrid(g);
	FLOAT *top = new FLOAT[grid_length];

	for (int i = 0; i < grid_length; i++)
		top[i] = asimptota;

	g2.data = top;

	return CalculateDirectProblem(g2, g, dsigma, mpi_rank, mpi_size);
}

FLOAT* CalculateDirectProblem(Grid& g, double dsigma, int mpi_rank, int mpi_size)
{
	double asimptota = g.get_Average();
	return CalculateDirectProblem(g, asimptota, dsigma, mpi_rank, mpi_size);
}
