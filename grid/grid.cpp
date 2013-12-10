#include "Grid.h"

extern "C" {
	bool ReadGrid(const char* fileName, double** data, int* nRow, int* nCol, double* xLL, double* yLL, double* xSize, double* ySize, double* blank)
	{
		Grid g(fileName);
		*data = g.data;
		*nRow = g.nRow;
		*nCol = g.nCol;
		*xLL = g.xLL;
		*yLL = g.yLL;
		*xSize = g.xSize;
		*ySize = g.ySize;
		*blank = g.BlankValue;
		
		return true;
	}

}