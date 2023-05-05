#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../global.h"
#include "../cuda/recalc.h"
#include "../cuda/info.h"

#include "../grid/Grid.h"

CUDA_FLOAT* CalculateRecalcUp(Grid& field, double height, int mpi_rank, int mpi_size)
{
	int mpi_rows_portion = field.nRow / mpi_size;
	printf("Every MPI thread will process %d rows\n", mpi_rows_portion);

	int grid_length = field.nCol * field.nRow;

	CUDA_FLOAT *result;
	if (mpi_rank == MPI_MASTER)
	{
		result = new CUDA_FLOAT[mpi_size * grid_length];
		memset(result, 0, mpi_size * grid_length * dsize);
	}
	else
	{
		result = new CUDA_FLOAT[grid_length];
		memset(result, 0, grid_length * dsize);
	}
	

	if (!Recalc(field.data, height, result, field.nCol, field.nRow, mpi_rows_portion * mpi_rank, mpi_rows_portion))
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
			result[j] /= 2 * MATH_PI;

		printf("%f\n", result[(field.nRow / 2) * field.nCol + field.nCol / 2]);
	
		return result;
	}
	
	return NULL;

}