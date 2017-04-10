#include <iostream>
using namespace std;

#include "global.h"
#include "grid/Grid.h"

#ifdef GEO_BUILD_M2KM

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: m_to_km <m.grd> <km.grd>" << endl;
		return 1;
	}
	
	Grid g(argv[1]);
	
	g.xLL /= 1000;
	g.yLL /= 1000;
	g.xSize /= 1000;
	g.ySize /= 1000;
	
	g.Write(argv[2]);
}

#endif