#include <stdio.h>
#include <math.h>

#include "global.h"
#include "cuda/Vz.h"

#include "fund.c"

int main()
{
	cudaPrintInfo();
	
	FLOAT *result = (FLOAT*)malloc(SIDE * SIDE * dsize);
	memset(result, 0, SIDE * SIDE * dsize);

	FLOAT *top= (FLOAT*)malloc(SIDE * SIDE * dsize);
	memset(top, 0, SIDE * SIDE * dsize);

	if (!CalculateVz(top, grid, result))
		return 1;

	printf("%f\n", result[(SIDE / 2) * SIDE + SIDE / 2]);
	
	FILE *f = fopen("d:\\fund_f.bin", "wb");
	fwrite(result, dsize, SIDE * SIDE, f);
	fclose(f);

	return 0;
}

