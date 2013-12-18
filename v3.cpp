#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>
#define MPI_MASTER 0

#include "global.h"
#include "cuda/Vz.h"
#include "cuda/info.h"

#include "grid/Grid.h"

const double GRAVITY_CONST = 6.67384; // CODATA

int main(int argc, char** argv)
{
	int mpi_rank, mpi_size;

	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */

	cudaPrintInfo();
	if (argc < 3)
	{
		printf("Usage: v3 <filename.grd> <density> [output.grd]\n");
		return 1;
	}
	char* filename = argv[1];
	double dsigma = atof(argv[2]);
	char* outputFilename = NULL;
	if (argc > 3)
		outputFilename = argv[3];

	Grid g(filename);

	printf("Grid read: %d x %d\n", g.nRow, g.nCol);

	if (g.nCol != g.nRow)
	{
		printf("Error: can not process non-square grid\n");
		MPI_Finalize();
		return 1;
	}
	
	if (g.nRow % mpi_size != 0)
	{
		printf("Error: nRow can not be divided to MPI thread count \n");
		MPI_Finalize();
		return 1;
	}
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

	FLOAT *top = new FLOAT[grid_length];
	double assimptota = g.get_Average();
	for (int i = 0; i < grid_length; i++)
		top[i] = assimptota;

	if (!CalculateVz(top, g.data, result, g.nCol, g.nRow, mpi_rows_portion * mpi_rank, mpi_rows_portion))
	{
		MPI_Abort(MPI_COMM_WORLD, 0);
		return 1;
	}

	if (mpi_rank != MPI_MASTER)
	{
		MPI_Send(result, grid_length, MPI_DOUBLE, MPI_MASTER, 0, MPI_COMM_WORLD);
	}
	else
	{
		for (int i = 1; i < mpi_size; i++)
		{
			MPI_Recv(&result[i * grid_length], grid_length, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		for (int i = 1; i < mpi_size; i++)
		{
			for (int j = 0; j < grid_length; j++)
				result[j] += result[i * grid_length + j];
		}

		for (int j = 0; j < grid_length; j++)
			result[j] *= GRAVITY_CONST * dsigma;
		
		printf("%f\n", result[(g.nRow / 2) * g.nCol + g.nCol / 2]);
	
		if (outputFilename != NULL)
		{
			g.data = result;
			g.zMin = g.get_Min();
			g.zMax = g.get_Max();
			g.Write(outputFilename);
		}
	}
	
	MPI_Finalize();
	
	return 0;
}

