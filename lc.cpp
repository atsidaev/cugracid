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

#ifdef GEO_BUILD_LC

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
	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif

	cudaPrintInfo();
	if (argc < 6)
	{
		printf("Usage: lc <observed_field.grd> <boundary.grd> <density> <alpha> <iterations> [output.grd]\n");
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
	
	fill_blank(modelBoundary);
	fill_blank(boundary);

	printf("Calculating c_function...");
	
	FLOAT* model_field = CalculateDirectProblem(boundary, dsigma, mpi_rank, mpi_size);
	for (int j = 0; j < boundary.nCol * boundary.nRow; j++)
		observedField.data[j] = observedField.data[j] - model_field[j];

	put_to_0(observedField.data, boundary.nCol * boundary.nRow);

	printf("Done!\n");
	
	printf("Field grid read\n");
	
	for (int i = 0; i < iterations; i++)
	{
		printf("Iteration %d\n", i);
		gridInfo(boundary);
		gridInfo(modelBoundary);
		gridInfo(observedField);
		FLOAT* result = CalculateDirectProblem(modelBoundary, boundary, dsigma, mpi_rank, mpi_size);
		
		//put_to_0(result, boundary.nCol * boundary.nRow);

		// printf("Result at 128, 128: %f\n", result[128 * 256 + 128]);

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
				sum += abs(observedField.data[j] - result[j]);
			
			//	if (boundary.data[j] > 0.5)
				{
					// boundary.data[j] /= (1 + alpha * boundary.data[j] * (observedField.data[j] - result[j]));
					//if (isnan(result[j]))
					//	printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
					boundary.data[j] /= (1 + boundary.data[j] * alpha * (observedField.data[j] - a * result[j]));
				}
			//	else
			//		boundary.data[j] = (observedField.data[j] - a * result[j]) / (2 * M_PI * GRAVITY_CONST * dsigma);
					
				
				/*double min = 4;
				double max = 26;
				if (boundary.data[j] < min)
					boundary.data[j] = min; 
				if (boundary.data[j] > max)
					boundary.data[j] = max;*/
			}
			printf("Deviation: %f\n", sum / (boundary.nCol * boundary.nRow));
			gridInfo(boundary);
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

#endif