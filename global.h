#define CUDA_FLOAT double
#define dsize sizeof(CUDA_FLOAT)

#define MPI_MASTER 0

#define MATH_PI 3.141592653589793238462643383279502884197169399375106

const double GRAVITY_CONST = 6.67384; // CODATA

int main_lc(int argc, char** argv);
int main_v3(int argc, char** argv);
int main_m2km(int argc, char** argv);
int main_compare(int argc, char** argv);