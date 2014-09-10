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

double* min_g1;
double* min_g2;
int min_items;

double minimized_function(double alpha)
{
	double sum = 0;
	for (int i = 0; i < min_items; i++)
	{
		sum += abs(min_g1[i] - alpha * min_g2[i]);
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
	
	for (int i = 0; i < iterations; i++)
	{
		printf("Iteration %d\n", i);
		FLOAT* result = CalculateDirectProblem(boundary, asimptota, dsigma, mpi_rank, mpi_size);

		printf("Result at 128, 128: %f\n", result[128 * 256 + 128]);

		if (mpi_rank == MPI_MASTER)
		{
			min_g1 = observedField.data;
			min_g2 = result;
			min_items = boundary.nCol * boundary.nRow;
		
			double a = 1;//golden_section(minimized_function, 0, 20, 30);
			printf("Calculated alpha: %f\n", a);
			
			double sum;
			
			for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
			{
				sum += abs(observedField.data[j] - result[j]);
			
				if (boundary.data[j] > 0.5)
				{
					// boundary.data[j] /= (1 + alpha * boundary.data[j] * (observedField.data[j] - result[j]));
					//if (isnan(result[j]))
					//	printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
					boundary.data[j] /= (1 + boundary.data[j] * alpha * (observedField.data[j] - a * result[j]));
				}
				else
					boundary.data[j] += (observedField.data[j] - a * result[j]) / (-2 * M_PI * GRAVITY_CONST * dsigma);
				
				/* double min =0;
				double max = 12;
				if (boundary.data[j] < min)
					boundary.data[j] = min; 
				if (boundary.data[j] > max)
					boundary.data[j] = max;  */
			}
			printf("Deviation: %f\n", sum / (boundary.nCol * boundary.nRow));
			printf("boudary: %f\n", boundary.data[128 * 256 + 128]);
			delete result;
		}
	
	}
	
	if (mpi_rank == MPI_MASTER)
	{
		boundary.zMin = boundary.get_Min();
		boundary.zMax = boundary.get_Max();
		boundary.Write(outputFilename);
	}

#ifdef USE_MPI	
	MPI_Finalize();
#endif	
	
	return 0;
}

