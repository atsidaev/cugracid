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

#include "direct.h"

#ifdef GEO_BUILD_V3

int main(int argc, char** argv)
{
	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif

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
	fill_blank(g);
	FLOAT* result = CalculateDirectProblem(g, dsigma, mpi_rank, mpi_size);

	if (mpi_rank == MPI_MASTER && outputFilename != NULL)
	{
		g.data = result;
		g.zMin = g.get_Min();
		g.zMax = g.get_Max();
		g.Write(outputFilename);
	}

#ifdef USE_MPI
	MPI_Finalize();
#endif	
	return 0;
}

#endif