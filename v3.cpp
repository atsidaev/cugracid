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

#include "file_utils.h"
#include "grid/Grid.h"

#include "direct.h"

#ifdef WIN32
// TODO: split getopt.h to .h and .c
#include "windows/getopt_header.h"
#else
#include <unistd.h>
#include <getopt.h>
#endif

int main_v3(int argc, char** argv)
{
	static struct option long_options[] =
	{
		{ "boundary", required_argument, NULL, 'b' },	// initial boundary file name
		{ "dsigma", required_argument, NULL, 's' },		// delta sigma value
		{ "asimptota", required_argument, NULL, 't' },	// depth of selected asimptita plane
		{ "output", required_argument, NULL, 'o' },		// output boundary grid file name
		{ "help", required_argument, NULL, 'h' },		// output boundary grid file name
		{ NULL, 0, NULL, 0 }
	};

	char* filename = NULL;
	double dsigma = NAN;
	char* dsigmaFileName = NULL;
	double asimptota = NAN;
	char* outputFilename = NULL;
	char print_help = 0;

	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "b:s:t:o:h:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
			case 'b':
				filename = optarg; break;
			case 's':
			{
				std::ifstream f(optarg);
				if (f.good())
					dsigmaFileName = optarg;
				else
					dsigma = atof(optarg);
				break;
			}
			case 'o':
				outputFilename = optarg; break;
			case 't':
				asimptota = atof(optarg); break;
			case 'h':
				print_help = 1; break;
				/* Long-only options */
			default:
				fprintf(stderr, "Invalid argument %c\n", c);
				return 1;
		}
	}

	if (print_help || argc == 1)
	{
		option* o = long_options;
		fprintf(stderr, "Program for gravity field forward calculation on CUDA GPU. (C) Alexander Tsidaev, 2014-2019\nValid options:\n");
		while (o->name != NULL)
		{
			fprintf(stderr, "\t--%s", o->name);
			if (o->val != 0)
				fprintf(stderr, ", -%c", o->val);
			fprintf(stderr, "\n");
			o++;
		}
		return 255;
	}

	if (filename == NULL)
	{
		fprintf(stderr, "Boundary grid is not specified\n");
		return 1;
	}

	if (!file_exists(filename))
	{
		fprintf(stderr, "Boundary grid file does not exist\n");
		return 1;
	}

	cudaPrintInfo();

	// Begin calculation

	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init(&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif


	Grid g(filename);
	printf("Grid read: %d x %d\n", g.nRow, g.nCol);
	fill_blank(g);
	CUDA_FLOAT* result;

	asimptota = isnan(asimptota) ? g.get_Average() : asimptota;

	if (dsigmaFileName == NULL)
		result = CalculateDirectProblem(g, asimptota, dsigma, NULL, mpi_rank, mpi_size);
	else
	{
		Grid dsigmaFile(dsigmaFileName);
		result = CalculateDirectProblem(g, asimptota, 0, &dsigmaFile, mpi_rank, mpi_size);
	}

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
