#include <cstring>
#include <cstdlib>
#include <cerrno>

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

	char * e;
	errno = 0;

	char* filename = argv[1];
	double dsigma = std::strtod(argv[2], &e);
	Grid* dsigmaGrid = NULL;

	if (*e != '\0' || errno != 0) // str to double conversion failed
	{
		ifstream f(argv[2]);
		if (!f.good())
		{
			std::cerr << "Invalid density value. Please provide constsnt double of grid file name" << endl;
			return 1;
		}
		dsigmaGrid = new Grid(argv[2]);
	}

	char* outputFilename = NULL;
	if (argc > 3)
		outputFilename = argv[3];

	Grid g(filename);
	printf("Grid read: %d x %d\n", g.nRow, g.nCol);
	fill_blank(g);
	CUDA_FLOAT* result;
	
	if (dsigmaGrid == NULL)
		result = CalculateDirectProblem(g, dsigma, mpi_rank, mpi_size);
	else
		result = CalculateDirectProblem(g, *dsigmaGrid, mpi_rank, mpi_size);

	if (mpi_rank == MPI_MASTER && outputFilename != NULL)
	{
		g.data = result;
		g.zMin = g.get_Min();
		g.zMax = g.get_Max();
		g.Write(outputFilename);
	}

	if (dsigmaGrid != NULL)
		delete dsigmaGrid;

#ifdef USE_MPI
	MPI_Finalize();
#endif	
	return 0;
}

#endif