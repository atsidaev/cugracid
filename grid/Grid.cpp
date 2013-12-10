#include "Grid.h"

__int32 Grid::ReadInt32(ifstream* fs)
{
	__int32 result;
	fs->read( reinterpret_cast<char*>(&result), sizeof result );
	return result;
}

double Grid::ReadDouble(ifstream* fs)
{
	double result;
	fs->read( reinterpret_cast<char*>(&result), sizeof result );
	return result;
}

bool Grid::Init()
{
	data = NULL;
	nRow = 0;
	nCol = 0;
	xLL = 0;
	yLL = 0;
	xSize = 0;
	ySize = 0;
	zMin = 0;
	zMax = 0;
	Rotation = 0;
	BlankValue = 0;
}

Grid::Grid()
{
	Init();
}

Grid::Grid(const char* fileName)
{
	Read(fileName);
}

bool Grid::Read(const char* fileName)
{
	ifstream ifs(fileName);

	while (true)
	{
		int ID = ReadInt32(&ifs);
		if (ID == 0x42525344)	// header DSRB
		{
			int Size = ReadInt32(&ifs);
			int Version = ReadInt32(&ifs);
			continue;
		}
		if (ID == 0x44495247)	// grid GRID
		{
			int Size = ReadInt32(&ifs);

			nRow = ReadInt32(&ifs);
			nCol = ReadInt32(&ifs);
			xLL = ReadDouble(&ifs);
			yLL = ReadDouble(&ifs);
			xSize = ReadDouble(&ifs);
			ySize = ReadDouble(&ifs);
			zMin = ReadDouble(&ifs);
			zMax = ReadDouble(&ifs);
			Rotation = ReadDouble(&ifs);
			BlankValue = ReadDouble(&ifs);

			data = new double[nCol * nRow];
			continue;
		}
		if (ID == 0x41544144)	// data
		{
			int Size = ReadInt32(&ifs);
			for (int i = 0; i < nRow; i++)
				for (int j = 0; j < nCol; j++)
					data[i * nCol + j] = ReadDouble(&ifs);
				
			break;
		}
		if (ID == 0x49544c46)	// fault
		{
			ifs.close();
			return false;
		}
	}

	ifs.close();
	return true;
}
