#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "global.h"
#include "cuda/Vz.h"
#include "cuda/info.h"

#include "grid/Grid.h"

#include "direct.h"

int main(int argc, char** argv)
{
	int mpi_rank, mpi_size;

	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */

	cudaPrintInfo();
	if (argc < 6)
	{
		printf("Usage: lc <field.grd> <boundary.grd> <density> <alpha> <iterations> [output.grd]\n");
		return 1;
	}
	char* fieldFilename = argv[1];
	char* boundaryFilename = argv[2];
	double dsigma = atof(argv[3]);
	double alpha = atof(argv[4]);
	double iterations = atoi(argv[5]);
	
	char* outputFilename = NULL;
	if (argc > 6)
		outputFilename = argv[6];

	Grid observedField(fieldFilename);
	Grid modelBoundary(boundaryFilename);

	Grid boundary(boundaryFilename);
	
	printf("Field grid read\n");
	
	for (int i = 0; i < iterations; i++)
	{
		printf("Iteration %d\n", i);
		FLOAT* result = CalculateDirectProblem(boundary, modelBoundary, dsigma, mpi_rank, mpi_size);

		printf("Result at 128, 128: %f\n", result[128 * 256 + 128]);

		if (mpi_rank == MPI_MASTER)
		{
			for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
				boundary.data[j] /= (1 + alpha * boundary.data[j] * (observedField.data[j] - result[j]));
		
			delete result;
		}
	
	}
	
	if (mpi_rank == MPI_MASTER)
	{
		boundary.zMin = boundary.get_Min();
		boundary.zMax = boundary.get_Max();
		boundary.Write(outputFilename);
	}
	
	MPI_Finalize();
	
	
	return 0;
}

