#include <cstring>
#include <fstream>

#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../global.h"
#include "../cuda/info.h"

#include "../grid/Grid.h"

#include "../calc/direct.h"

#include "../file_utils.h"

#ifdef WIN32
#include "../windows/getopt.h"
#else
#include <unistd.h>
#include <getopt.h>
#endif

typedef enum { IDT_BOUNDARY, IDT_ASIMPTOTIC_PLANE } initial_data_type_t;
typedef enum { EC_EPSILON, EC_ITERATIONS_NUMBER } exit_contition_t;

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

void DebugGridSave(char* fileNamePrefix, int iteration, Grid& grid)
{
	char buf[256];
	sprintf(buf, "%s_%04d.grd", fileNamePrefix, iteration);
	grid.Write((const char*)buf);
}

void DebugGridSave(char* fileNamePrefix, int iteration, double* data, Grid& size)
{
	auto grid = Grid::GenerateEmptyGrid(size);
	for (int i = 0; i < grid.nCol * grid.nRow; i++)
			grid.data[i] = data[i];

  char buf[256];
	sprintf(buf, "%s_%04d.grd", fileNamePrefix, iteration);
	grid.Write((const char*)buf);
}

int main_lc(int argc, char** argv)
{
	char* fieldFilename = NULL;
	char* outputFilename = NULL;
	char* initialBoundaryFileName = NULL;
	char* dsigmaFileName = NULL;

	char* outFieldPrefix = NULL;
	char* outDiffFieldPrefix = NULL;
	char* outSurfacePrefix = NULL;
	char* outDiffSurfacePrefix = NULL;

	double dsigma = NAN;
	char* dsigmaFile = NULL;
	double alpha = NAN;
	double epsilon = NAN;
	double asimptota = NAN;
	int iterations = 0;
	int quit_after_diverged_iterations = 0;
	char near_surface = 0;
	char print_help = 0;

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
		{ "quit-after-diverged-iterations", required_argument, NULL, 'q' },		// stop process if it diverges for N steps
		{ "help", no_argument, NULL, 'h' },				// help 
		{ "near-surface", no_argument, NULL, 'n' },			// use near-surface formula for calculation
		{ "out-field-prefix", required_argument, NULL, 0 },		// field debug output on each iteration
		{ "out-diff-field-prefix", required_argument, NULL, 0 },	// diff (U-Un) debug output on each iteration
		{ "out-surface-prefix", required_argument, NULL, 0 },		// surface output on each iteration file prefix
		{ "out-diff-surface-prefix", required_argument, NULL, 0 },		// diff surface output on each iteration file prefix
		{ NULL, 0, NULL, 0 }
	};

	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "f:s:b:a:o:e:i:t:q:nh", long_options, &option_index)) != -1)
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
			asimptota = atof(optarg); break;
		case 'q':
			quit_after_diverged_iterations = atoi(optarg); break;
		case 'n':
			near_surface = 1; break;
		case 'h':
			print_help = 1; break;
		/* Long-only options */
		case 0:
			if (strcmp(long_options[option_index].name, "out-field-prefix") == 0)
				outFieldPrefix = optarg;
			else if (strcmp(long_options[option_index].name, "out-diff-field-prefix") == 0)
				outDiffFieldPrefix = optarg;
			else if (strcmp(long_options[option_index].name, "out-surface-prefix") == 0)
				outSurfacePrefix = optarg;
			else if (strcmp(long_options[option_index].name, "out-diff-surface-prefix") == 0)
				outDiffSurfacePrefix = optarg;

			break;
		default:
			fprintf(stderr, "Invalid argument %c\n", c);
			return 1;
		}
	}

	if (print_help || argc == 1)
	{
		option* o = long_options;
		fprintf(stderr, "Program for Local Corrections calculation on CUDA GPU. (C) Alexander Tsidaev, 2014-2021\nValid options:\n");
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

	if (fieldFilename == NULL)
	{
		fprintf(stderr, "Field should be specified\n");
		return 1;
	}

	if (isnan(epsilon) && iterations == 0)
	{
		fprintf(stderr, "One of arguments -e should be specified if -i omitted\n");
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

	initial_data_type_t initial_data_type = initialBoundaryFileName == NULL ? IDT_ASIMPTOTIC_PLANE : IDT_BOUNDARY;
	exit_contition_t exit_condition = isnan(epsilon) ? EC_ITERATIONS_NUMBER : EC_EPSILON;

	int mpi_rank = 0, mpi_size = 1;

#ifdef USE_MPI
	MPI_Init (&argc, &argv);      /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);        /* get number of processes */
#endif
	std::vector<unsigned char> devices_list;
	cudaPrintInfo(devices_list);

	if (!file_exists(fieldFilename) || !file_exists(fieldFilename))
		return 1;

	Grid observedField(fieldFilename);

	Grid* dsigmaGrid = NULL;
	if (dsigmaFile != NULL)
	{
		if (!file_exists(dsigmaFile))
			return 1;

		dsigmaGrid = new Grid(dsigmaFile);
	}

	// Set z0 boundary to asimptota value or read initial boundary file
	Grid* z0 = NULL;
	if (initial_data_type == IDT_BOUNDARY)
	{
		if (!file_exists(initialBoundaryFileName))
			return 1;

		z0 = new Grid(initialBoundaryFileName);
		fill_blank(*z0);
	}
	else
	{
		z0 = new Grid(observedField);
		create_empty_data(*z0);
		fill_with_value(*z0, asimptota);
	}

	double nCol = z0->nCol, nRow = z0->nRow;

	// Create asimptota grid from z0 (real asimptota or average value of initial boundary position)
	Grid asimptotaGrid(*z0);
	create_empty_data(asimptotaGrid);
	if (initial_data_type == IDT_BOUNDARY && !isnan(asimptota))
		fill_with_value(asimptotaGrid, asimptota);
	else
		fill_with_value(asimptotaGrid, z0->get_Average());

	// Set observed field to 0
	auto avg_U0 = observedField.get_Average();
	put_to_0(observedField.data, observedField.nCol * observedField.nRow);
	double rms_U0 = get_Rms(observedField);
	printf("avg(U0)=%f (was set to 0 then), rms(U0)=%f\n", avg_U0, rms_U0);

	// We restore only an addition to the field, so removing model field from the observed one
	printf("Calculating model field...");
	CUDA_FLOAT* f_model;
	f_model = CalculateDirectProblem(asimptotaGrid, *z0, dsigma, dsigmaGrid, mpi_rank, mpi_size, devices_list);
	put_to_0(f_model, observedField.nCol * observedField.nRow);
	// TODO: need to be tested since it produces invalid result for asymptote-based problem
	// for (int j = 0; j < nCol * nRow; j++)
	// 	observedField.data[j] -= f_model[j];
	printf("Done!\n");

	Grid z_n(*z0);
	create_empty_data(z_n);
	copy_data(z_n, *z0);

	double prev_rms_f_div_rms_U0 = 0;
	int diverged_iterations = 0;

	// Main calculation loop
	for (int iteration = 0; iterations > 0 && iteration < iterations || iterations == 0; iteration++)
	{
		printf("Iteration %d: ", iteration);

		CUDA_FLOAT* result;
		result = CalculateDirectProblem(asimptotaGrid, z_n, dsigma, dsigmaGrid, mpi_rank, mpi_size, devices_list);

		if (mpi_rank == MPI_MASTER)
		{
			if (outFieldPrefix != NULL)
				DebugGridSave(outFieldPrefix, iteration, result, z_n);

			double rms_f = 0, sum_f = 0, sum_g = 0, avg_Un = 0;

			for (int j = 0; j < z_n.nCol * z_n.nRow; j++)
			{
				auto diffU = observedField.data[j] - result[j];
				double b;
				if (!near_surface)
					b = z_n.data[j] / (1 + alpha * z_n.data[j] * diffU);
				else
					b = z_n.data[j] - alpha * (diffU / (2 * MATH_PI * GRAVITY_CONST));
				
				rms_f += diffU * diffU;
				sum_f += diffU;
				sum_g += z_n.data[j] - b;
				avg_Un += result[j];

				z_n.data[j] = b;
			}

			avg_Un /= nCol * nRow;

			rms_f = sqrt(rms_f / (nCol * nRow));
			double avgZ = z_n.get_Average();

			double rms_Z = 0, rms_Un = 0;
			for (int j = 0; j < nCol * nRow; j++)
			{
				auto z_diff = z_n.data[j] - avgZ;
				auto u_diff = result[j] - avg_Un;
				rms_Z += z_diff * z_diff;
				rms_Un += u_diff * u_diff;
			}
			rms_Z = sqrt(rms_Z / (nCol * nRow));
			rms_Un = sqrt(rms_Un / (nCol * nRow));

			if (outSurfacePrefix != NULL)
				DebugGridSave(outSurfacePrefix, iteration, z_n);

			if (outDiffSurfacePrefix != NULL)
			{
				auto diff = Grid::Diff(observedField, result);
				DebugGridSave(outDiffSurfacePrefix, iteration, diff);
			}

			delete result;

			auto deviation_f = sum_f / (nCol * nRow);
			auto deviation_g = sum_g / (nCol * nRow);
			auto rms_f_div_rms_U0 = rms_f / rms_U0;
			printf("avg(U-Un)=%f \trms(U-Un)/rms(U)=%f \tavg(Zn+1 - Zn)=%f\tavg(Zn+1)=%f,\trms(Zn+1 - avg(Zn+1))=%f,\trms(Un - avg(Un))=%f\n", deviation_f, rms_f_div_rms_U0, deviation_g, avgZ, rms_Z, rms_Un);
			
			if (exit_condition == EC_EPSILON && rms_f_div_rms_U0 < epsilon)
			{
				printf("Deviation %f is less than required epsilon %f, exiting. Iterations count %d.\n", rms_f_div_rms_U0, epsilon, iteration);
				break;
			}

			if (prev_rms_f_div_rms_U0 < rms_f_div_rms_U0)
			{
				diverged_iterations++;
				prev_rms_f_div_rms_U0 = rms_f_div_rms_U0;
				if (quit_after_diverged_iterations != 0 && iteration != 0 && quit_after_diverged_iterations == diverged_iterations)
				{
					printf("Diverged for %i iterations, exiting.\n", diverged_iterations);
					break;
				}
			}
			else
				diverged_iterations = 0;
		}

	}

	if (mpi_rank == MPI_MASTER)
	{
		if (outputFilename != NULL)
		{
			Grid boundary(z_n);
			create_empty_data(boundary);
			copy_data(boundary, z_n);
			//for (int j = 0; j < nCol * nRow; j++)
			//	boundary.data[j] += z0->data[j];

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
