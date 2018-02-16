#include <cstring>
#include <fstream>

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

#include "file_utils.h"
#include "golden.h"

#ifdef WIN32
#include "windows/getopt.h"
#else
#include <unistd.h>
#include <getopt.h>
#endif

#if defined(GEO_BUILD_LC)

typedef enum { IDT_BOUNDARY, IDT_ASIMPTOTIC_PLANE } initial_data_type_t;
typedef enum { EC_EPSILON, EC_ITERATIONS_NUMBER } exit_contition_t;

double* min_g1;
double* min_g2;
int min_items;

double minimized_function(double alpha)
{
	double sum = 0;
	for (int i = 0; i < min_items; i++)
	{
		sum += fabs(min_g1[i] - alpha * min_g2[i]);
	}
	double res = sum / min_items;
	// printf("	Min: for alpha=%f is %f\n", alpha, res);
	
	return res;
}

void gridInfo(Grid& boundary)
{
	printf("Grid info: %dx%d, X: %f...%f, Y: %f..%f, xSize: %f, ySize: %f, Min: %f, Max: %f\n", 
		boundary.nCol, boundary.nRow,
		boundary.xLL, boundary.xLL + (boundary.nCol - 1) * boundary.xSize,
		boundary.yLL, boundary.yLL + (boundary.nRow - 1) * boundary.ySize,
		boundary.xSize, boundary.ySize,
		boundary.get_Min(), boundary.get_Max());
}

void put_to_0(double* result, int size)
{
		double avg = 0;
		for (int j = 0; j < size; j++)
			avg += result[j];
		avg /= size;
		
		for (int j = 0; j < size; j++)
			result[j] -= avg;
}

int main(int argc, char** argv)
{
	char* fieldFilename = NULL;
	char* outputFilename = NULL;
	char* initialBoundaryFileName = NULL;
	char* dsigmaFileName = NULL;

	double dsigma = NAN;
	char* dsigmaFile = NULL;
	double alpha = NAN;
	double epsilon = NAN;
	double asimptota = NAN;
	int iterations = 0;


	static struct option long_options[] =
	{
		{ "field", required_argument, NULL, 'f' },		// field file name
		{ "dsigma", required_argument, NULL, 's' },		// delta sigma value
		{ "boundary", required_argument, NULL, 'b' },	// initial boundary file name
		{ "alpha", required_argument, NULL, 'a' },		// alpha stabilizer value
		{ "eps", required_argument, NULL, 'e' },		// epsilon max discrepancy value
		{ "iterations", required_argument, NULL, 'i' },	// max count of required iterations
		{ "asimptota", required_argument, NULL, 't' },	// depth of selected asimptita plane
		{ "output", required_argument, NULL, 'o' },		// output boundary grid file name
		{ NULL, 0, NULL, 0 }
	};

	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "f:s:b:a:o:e:i:t:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
		case 'f':
			fieldFilename = optarg; break;
		case 's':
		{
			std::ifstream f(optarg);
			if (f.good())
				dsigmaFile = optarg;
			else
				dsigma = atof(optarg); 
			break;
		}
		case 'b':
			initialBoundaryFileName = optarg; break;
		case 'a':
			alpha = atof(optarg); break;
		case 'o':
			outputFilename = optarg; break;
		case 'e':
			epsilon = atof(optarg); break;
		case 'i':
			iterations = atoi(optarg); break;
		case 't':
			asimptota = atoi(optarg); break;
		default:
			fprintf(stderr, "Invalid argument\n");
			return 1;
		}
	}

	if (fieldFilename == NULL)
	{
		fprintf(stderr, "Field should be specified\n");
		return 1;
	}

	if (!(iterations == 0 ^ isnan(epsilon)))
	{
		fprintf(stderr, "One of arguments -i or -e should be specified\n");
		return 1;
	}

	if (isnan(alpha))
	{
		fprintf(stderr, "Alpha should be specified\n");
		return 1;
	}

	if (isnan(dsigma) && dsigmaFile == NULL)
	{
		fprintf(stderr, "Delta sigma should be specified\n");
		return 1;
	}

	if (!(isnan(asimptota) ^ initialBoundaryFileName == NULL))
	{
		fprintf(stderr, "One of arguments -t or -b should be specified\n");
		return 1;
	}

	initial_data_type_t initial_data_type = initialBoundaryFileName == NULL ? IDT_ASIMPTOTIC_PLANE : IDT_BOUNDARY;
	exit_contition_t exit_condition = isnan(epsilon) ? EC_ITERATIONS_NUMBER : EC_EPSILON;

	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif

	cudaPrintInfo();

	if (!file_exists(fieldFilename) || !file_exists(fieldFilename))
		return 1;

	Grid observedField(fieldFilename);
	
	Grid boundary(fieldFilename); // z_n

	Grid* dsigmaGrid = NULL;
	if (dsigmaFile != NULL)
	{
		if (!file_exists(dsigmaFile))
			return 1;

		dsigmaGrid = new Grid(dsigmaFile);
	}

	// Set boundary to asimptota value or read initial boundary file
	Grid* modelBoundary = NULL;
	if (initial_data_type == IDT_BOUNDARY)
	{
		modelBoundary = new Grid(initialBoundaryFileName);
		fill_blank(*modelBoundary);
		
		boundary.Read(initialBoundaryFileName);
		fill_blank(boundary);
	}
	else
	{
		for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
			boundary.data[j] = asimptota;
	}

	printf("Calculating c_function...");
	
	CUDA_FLOAT* model_field = CalculateDirectProblem(boundary, dsigma, dsigmaGrid, mpi_rank, mpi_size);
	for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
		observedField.data[j] = observedField.data[j] - model_field[j];

	put_to_0(observedField.data, boundary.nCol * boundary.nRow);

	printf("Done!\n");
	
	printf("Field grid read\n");
	
	for (int i = 0; exit_condition == EC_EPSILON || i < iterations; i++)
	{
		printf("Iteration %d\n", i);
		// gridInfo(boundary);
		// gridInfo(modelBoundary);
		// gridInfo(observedField);
		CUDA_FLOAT* result;
		if (modelBoundary != NULL)
			result = CalculateDirectProblem(*modelBoundary, boundary, dsigma, dsigmaGrid, mpi_rank, mpi_size);
		else
			result = CalculateDirectProblem(boundary, asimptota, dsigma, dsigmaGrid, mpi_rank, mpi_size);
		
		//put_to_0(result, boundary.nCol * boundary.nRow);

		// printf("Result at 128, 128: %f\n", result[128 * 256 + 128]);

		if (mpi_rank == MPI_MASTER)
		{
			min_g1 = observedField.data;
			min_g2 = result;
			min_items = boundary.nCol * boundary.nRow;
		
			double sum_f = 0, sum_g = 0;
			
			for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
			{
				auto b = boundary.data[j] / (1 + boundary.data[j] * alpha * (observedField.data[j] - result[j]));

				sum_f += fabs(observedField.data[j] - result[j]);
				sum_g += fabs(boundary.data[j] - b);

				boundary.data[j] = b;
			}
			delete result;

			auto deviation_f = sum_f / (boundary.nCol * boundary.nRow);
			auto deviation_g = sum_g / (boundary.nCol * boundary.nRow);
			printf("Deviation: field: %f, grid: %f\n", deviation_f, deviation_g);
			if (exit_condition == EC_EPSILON && deviation_f < epsilon)
			{
				printf("Deviation is less than required epsilon %f, exiting. Iteration count %d.\n", epsilon, i);
				break;
			}
			//gridInfo(boundary);
		}
	
	}
	
	if (mpi_rank == MPI_MASTER)
	{
		if (outputFilename != NULL)
		{
			boundary.zMin = boundary.get_Min();
			boundary.zMax = boundary.get_Max();
			boundary.Write(outputFilename);
		}
		else
			fprintf(stderr, "Warning: Output file is not specified\n");
	}

#ifdef USE_MPI	
	MPI_Finalize();
#endif
	
	return 0;
}

#endif
