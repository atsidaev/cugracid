#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <tuple>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../global.h"
#include "../cuda/Vz.h"
#include "../cuda/info.h"

#include "../file_utils.h"
#include "../grid/Grid.h"

#include "../calc/direct.h"

#ifdef WIN32
// TODO: split getopt.h to .h and .c
#include "windows/getopt_header.h"
#else
#include <unistd.h>
#include <getopt.h>
#endif

const char* TEST = "test";
bool run_test = 0;
void test_vz();

int main_vz(int argc, char** argv)
{
	static struct option long_options[] =
	{
		{ "boundary", required_argument, NULL, 'b' },   // initial boundary file name
		{ "dsigma", required_argument, NULL, 's' },     // delta sigma value
		{ "asimptota", required_argument, NULL, 't' },  // depth of selected asimptotic plane
		{ "output", required_argument, NULL, 'o' },     // output boundary grid file name
		{ "devices", required_argument, NULL, 'd' },    // which devices to use
		{ "print", required_argument, NULL, 'p' },      // print value at point at the end
		{ "help", required_argument, NULL, 'h' },       // output boundary grid file name
		{ TEST, 0, NULL, 0},                            // output boundary grid file name
		{ NULL, 0, NULL, 0 }
	};

	char* filename = NULL;
	double dsigma = NAN;
	char* dsigmaFileName = NULL;
	double asimptota = NAN;
	char* outputFilename = NULL;
	char print_help = 0;

	std::vector<unsigned char> devices_list;
	std::vector<std::tuple<unsigned int, unsigned int>> monitoring_points;

	bool monitoring_points_valid = true;

	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "b:s:t:o:h:d:p:", long_options, &option_index)) != -1)
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
			case 'd':
			{
				char* pos;
				while ((pos = strchr(optarg, ',')) != NULL) {
					*pos = 0;
					devices_list.push_back(atof(optarg));
					optarg = pos + 1;
				}
				devices_list.push_back(atof(optarg));
				break;
			}
			case 'p':
			{
				char* pos = optarg - 1;
				do
				{
					optarg = pos + 1;
					pos = strchr(optarg, ',');
					*pos = 0;
					int print_at_x = atoi(optarg);
					optarg = pos + 1;
					if (pos == NULL)
					{
						monitoring_points_valid = false;
						break;
					}
					int print_at_y = atoi(optarg);
					monitoring_points.push_back(std::tuple<int, int>(print_at_x, print_at_y));
					pos = strchr(optarg, '|');
				} while (pos != NULL);
				break;
			}
			default:
				if (strcmp(long_options[option_index].name, TEST) == 0)
					run_test = 1;
				else
				{
					fprintf(stderr, "Invalid argument %c\n", c);
					return 1;
				}
		}
	}

	if (print_help || argc == 1)
	{
		option* o = long_options;
		fprintf(stderr, "Program for gravity field forward calculation on CUDA GPU. (C) Alexander Tsidaev, 2014-2020\nValid options:\n");
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

	if (run_test)
	{
		test_vz();
		return 0;
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

	if (!monitoring_points_valid)
	{
		fprintf(stderr, "--print option should be 'x1,y1|x2,y2|...|xn,yn'\n");
		return 1;
	}

	if (devices_list.size() == 0)
		devices_list = getGpuDevices();

	cudaPrintInfo(devices_list);

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
		result = CalculateDirectProblem(g, asimptota, dsigma, NULL, mpi_rank, mpi_size, devices_list);
	else
	{
		Grid dsigmaFile(dsigmaFileName);
		result = CalculateDirectProblem(g, asimptota, 0, &dsigmaFile, mpi_rank, mpi_size, devices_list);
	}

	if (mpi_rank == MPI_MASTER)
	{
		if (monitoring_points_valid)
		{
			for (auto& element : monitoring_points)
			{
				auto print_at_x = std::get<0>(element);
				auto print_at_y = std::get<1>(element);
				printf("Value at (%d,%d) = %f\n", print_at_x, print_at_y, result[print_at_y * g.nCol + print_at_x]);
			}
		}

		if (outputFilename != NULL)
		{
			g.data = result;
			g.zMin = g.get_Min();
			g.zMax = g.get_Max();
			g.Write(outputFilename);
		}
	}

#ifdef USE_MPI
	MPI_Finalize();
#endif	
	return 0;
}

void debug_exit(double* grid, int nCol, int nRow)
{
	printf("Error! Debug output below\n");
	
	for (int j = 0; j < nRow; j++)
	{
		for (int i = 0; i < nCol; i++)
		{
			printf("%.2f ", grid[j * nCol + i]);
		}
		printf("\n");
	}
	exit(1);
}

void test_vz()
{
	const auto nCol = 16;
	const auto nRow = 16;

	Grid g = Grid();
	g.nCol = nCol;
	g.nRow = nRow;
	g.xLL = 0;
	g.yLL = 0;
	g.xSize = 10;
	g.ySize = 10;
	g.data = new double[nCol * nRow];
	fill_with_value(g, 1);

	double asimptota = 2;
	double dsigma = 1;

	auto devices_list = getGpuDevices();
	cudaPrintInfo(devices_list);

	while (devices_list.size() > 0)
	{
		printf("Testing with %ld GPU device(s)\n", devices_list.size());
		auto result = CalculateDirectProblem(g, asimptota, dsigma, NULL, MPI_MASTER, 1, devices_list);
		
		auto middle =  result[nCol * (nRow + 1) / 2];
		auto middleOk = middle > 41 && middle < 42;
		printf("Value at center (%f) is between 41 and 42: %s\n", middle, middleOk ? "YES" : "NO");
		if (!middleOk)
			debug_exit(result, nCol, nRow);

		bool symmetry = true;
		for (int j = 0; j < nRow / 2; j++)
			for (int i = 0; i < nCol / 2; i++)
			{
				auto v1 = result[j * nCol + i];
				auto v2 = result[(nRow - j - 1) * nCol + i];
				auto v3 = result[(nRow - j - 1) * nCol + (nCol - i - 1)];
				auto v4 = result[j * nCol + (nCol - i - 1)];
				symmetry &= (fabs((v1 - v2) - (v3 - v4)) < 0.00000001);
			}
		printf("Full 4-side symmetry: %s\n", symmetry ? "YES" : "NO");
		if (!symmetry)
			debug_exit(result, nCol, nRow);

		devices_list.pop_back();
	}
}