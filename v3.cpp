#include <cstring>

#include <stdio.h>
#include <math.h>

#include "global.h"
#include "cuda/Vz.h"
#include "cuda/info.h"

#include "grid/Grid.h"

int main(int argc, char** argv)
{
	cudaPrintInfo();
	if (argc < 2)
	{
		printf("Usage: v3 <filename.grd> <output.grd>\n");
		return 1;
	}
	char* filename = argv[1];
	char* outputFilename = NULL;
	if (argc > 2)
		outputFilename = argv[2];
	Grid g(filename);

	printf("Grid read: %d x %d\n", g.nRow, g.nCol);
	if (g.nCol != g.nRow)
	{
		printf("Error: can not process non-square grid\n");
		return 1;
	}
	
	FLOAT *result = new FLOAT[SIDE * SIDE];
	memset(result, 0, SIDE * SIDE * dsize);

	FLOAT *top = new FLOAT[SIDE * SIDE];
	memset(top, 0, SIDE * SIDE * dsize);

	if (!CalculateVz(top, g.data, result))
		return 1;

	printf("%f\n", result[(SIDE / 2) * SIDE + SIDE / 2]);
	
	if (outputFilename != NULL)
	{
		g.data = result;
		g.Write(outputFilename);
	}
	
	return 0;
}

