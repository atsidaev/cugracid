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

#include "golden.h"

#ifdef GEO_BUILD_LCASS

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

int main(int argc, char** argv)
{
	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif

	cudaPrintInfo();
	if (argc < 6)
	{
		printf("Usage: lc <field.grd> <asimptota> <density> <alpha> <iterations> [output.grd]\n");
		return 1;
	}
	char* fieldFilename = argv[1];
	double asimptota = atof(argv[2]);
	double dsigma = atof(argv[3]);
	double alpha = atof(argv[4]);
	double iterations = atoi(argv[5]);
	
	/*double topValue = asimptota - 5;
	if (topValue <= 0)
		topValue = asimptota + 1;*/
	
	char* outputFilename = NULL;
	if (argc > 6)
		outputFilename = argv[6];

	Grid observedField(fieldFilename);
	
	printf("Field grid read\n");
	
	Grid boundary(fieldFilename);
	
	Grid asimptotaBoundary(fieldFilename);
	for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
	{
		boundary.data[j] = asimptota;
		asimptotaBoundary.data[j] = asimptota;
	}
	
	for (int i = 0; i < iterations; i++)
	{
		printf("Iteration %d\n", i);
		CUDA_FLOAT* result = CalculateDirectProblem(boundary, asimptotaBoundary, dsigma, mpi_rank, mpi_size);

		int center = (boundary.nCol / 2) * boundary.nRow + boundary.nRow / 2;

		printf("Result at 128, 128: %f\n", result[center]);

		if (mpi_rank == MPI_MASTER)
		{
			min_g1 = observedField.data;
			min_g2 = result;
			min_items = boundary.nCol * boundary.nRow;
		
			double a = 1;//golden_section(minimized_function, 0, 20, 30);
			printf("Calculated alpha: %f\n", a);
			
			double sum = 0;
			
			for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
			{
				sum += fabs(observedField.data[j] - result[j]);
			
				boundary.data[j] /= (1 + boundary.data[j] * alpha * (observedField.data[j] - a * result[j]));
				
				/* double min =0;
				double max = 12;
				if (boundary.data[j] < min)
					boundary.data[j] = min; 
				if (boundary.data[j] > max)
					boundary.data[j] = max;  */
			}
			printf("Deviation: %f\n", sum / (boundary.nCol * boundary.nRow));
			printf("boudary: %f\n", boundary.data[center]);
			delete result;
		}
	
	}
	
	if (mpi_rank == MPI_MASTER)
	{
		boundary.zMin = boundary.get_Min();
		boundary.zMax = boundary.get_Max();

		if (outputFilename != NULL)
			boundary.Write(outputFilename);
	}

#ifdef USE_MPI	
	MPI_Finalize();
#endif	
	
	return 0;
}

#endif