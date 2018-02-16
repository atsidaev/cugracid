#define CUDA_FLOAT double
#define dsize sizeof(CUDA_FLOAT)

#define MPI_MASTER 0

#define MATH_PI 3.141592653589793238462643383279502884197169399375106;

const double GRAVITY_CONST = 6.67384; // CODATA

// Build parameters
#define GEO_BUILD_V3
//#define GEO_BUILD_LC
//#define GEO_BUILD_LCASS
//#define GEO_BUILD_RECALC
//#define GEO_BUILD_M2KM

#ifndef WIN32
	#define GEO_BUILD_V3
	#define GEO_BUILD_LC
	#define GEO_BUILD_LCASS
	#define GEO_BUILD_RECALC
	#define GEO_BUILD_M2KM
#endif