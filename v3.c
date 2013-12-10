#include <stdio.h>
#include <math.h>

#include "global.h"
#include "cuda/Vz.h"

#include "grid/grid.h"

int main()
{
	cudaPrintInfo();
	char filename[] = "moxo.grd";
	double* data;
	double xLL, yLL, xSize, ySize, Blank;
	int nCol, nRow;
	ReadGrid(filename, &data, &nRow, &nCol, &xLL, &yLL, &xSize, &ySize, &Blank);
	printf("Grid read: %d x %d\n", nRow, nCol);
	if (nCol != nRow)
	{
		printf("Error: can not process non-square grid\n");
		return 1;
	}
	
	FLOAT *result = (FLOAT*)malloc(SIDE * SIDE * dsize);
	memset(result, 0, SIDE * SIDE * dsize);

	FLOAT *top= (FLOAT*)malloc(SIDE * SIDE * dsize);
	memset(top, 0, SIDE * SIDE * dsize);

	if (!CalculateVz(top, data, result))
		return 1;

	printf("%f\n", result[(SIDE / 2) * SIDE + SIDE / 2]);
	
//	FILE *f = fopen("d:\\fund_f.bin", "wb");
//	fwrite(result, dsize, SIDE * SIDE, f);
//	fclose(f);

	return 0;
}

