#include <iostream>
#include <fstream>
using namespace std;

inline bool file_exists(const char* file_name)
{
	ifstream f(file_name);
	if (!f.good())
	{
		std::cerr << "File " << file_name << " not found!" << endl;
		return 0;
	}
	return 1;
}
