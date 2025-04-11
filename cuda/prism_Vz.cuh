#ifndef PRISM_VZ_CUH
#define PRISM_VZ_CUH

#include "../global.h"

struct VzCalc
{
	__device__
	static CUDA_FLOAT Vz1(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT z, CUDA_FLOAT xi, CUDA_FLOAT nu, CUDA_FLOAT z1, CUDA_FLOAT z2)
	{
		CUDA_FLOAT x_dif = (xi - x);
		CUDA_FLOAT y_dif = (nu - y);
		CUDA_FLOAT z_dif2 = (z2 - z);
		CUDA_FLOAT z_dif1 = (z1 - z);

		CUDA_FLOAT R1 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif1 * z_dif1);
		CUDA_FLOAT R2 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif2 * z_dif2);

		return 
			-((nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
			(xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

			((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
			(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1)))));
	}

	__device__
	static CUDA_FLOAT Vz2(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT z, CUDA_FLOAT xi, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT z1, CUDA_FLOAT z2)
	{
		return Vz1(x, y, z, xi, y2, z1, z2) - Vz1(x, y, z, xi, y1, z1, z2);
	}

	// Vz3
	__device__
	static CUDA_FLOAT calc(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT z, CUDA_FLOAT x1, CUDA_FLOAT x2, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT z1, CUDA_FLOAT z2)
	{
		return Vz2(x, y, z, x2, y1, y2, z1, z2) - Vz2(x, y, z, x1, y1, y2, z1, z2);
	}
};

struct VzCalcSimplified {
	__device__
	static CUDA_FLOAT calc(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT z, CUDA_FLOAT x1, CUDA_FLOAT x2, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT z1, CUDA_FLOAT z2)
	{
		if ((x1-x) * (x1-x) + (y1-y) * (y1-y) > 16)
		{
				auto cx = x1 + (x2 - x1) / 2;
				auto cy = y1 + (y2 - y1) / 2;
				auto cz = z1 + (z2 - z1) / 2;
				auto vol = fabs(x2 - x1) * fabs(y2 - y1) * fabs(z2 - z1);
				auto r = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (cz) * (cz));
				return vol * fabs(cz) / (r*r*r);
		}

		return VzCalc::Vz2(x, y, z, x2, y1, y2, z1, z2) - VzCalc::Vz2(x, y, z, x1, y1, y2, z1, z2);
	}
};

#endif