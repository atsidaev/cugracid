#include <float.h>

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

void Grid::WriteInt32(ofstream* fs, __int32 value)
{
	fs->write(reinterpret_cast<char*>(&value), sizeof value);
}

void Grid::WriteDouble(ofstream* fs, double value)
{
	fs->write(reinterpret_cast<char*>(&value), sizeof value);
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
			ifs.read((char*)data, nCol * nRow * sizeof(double));
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

bool Grid::Write(const char* fileName)
{
	ofstream ofs(fileName, ios::binary | ios::out);
	
	WriteInt32(&ofs, 0x42525344); // header DSRB
	WriteInt32(&ofs, sizeof(__int32));
	WriteInt32(&ofs, 2); // Version
	
	WriteInt32(&ofs, 0x44495247); // grid GRID
	WriteInt32(&ofs, 2 * sizeof(__int32) + 8 * sizeof(double));

	WriteInt32(&ofs, nRow);
	WriteInt32(&ofs, nCol);
	WriteDouble(&ofs, xLL);
	WriteDouble(&ofs, yLL);
	WriteDouble(&ofs, xSize);
	WriteDouble(&ofs, ySize);
	WriteDouble(&ofs, zMin);
	WriteDouble(&ofs, zMax);
	WriteDouble(&ofs, Rotation);
	WriteDouble(&ofs, BlankValue);

	WriteInt32(&ofs, 0x41544144); // data
	__int32 size = nCol * nRow * sizeof(double);
	WriteInt32(&ofs, size);
	ofs.write((char*)data, size);
	
	return true;
}

double Grid::get_Average()
{
	int count = 0;
	double sum = 0;
	for (int i = 0; i < nCol * nRow; i++)
		if (data[i] != BlankValue)
		{
			count++;
			sum += data[i];
		}
	
	return sum / count;
}

double Grid::get_Min()
{
	double min = DBL_MAX;
	for (int i = 0; i < nCol * nRow; i++)
		if (data[i] < min)
			min = data[i];
	return min;
}

double Grid::get_Max()
{
	double max = DBL_MIN;
	for (int i = 0; i < nCol * nRow; i++)
		if (data[i] > max)
			max = data[i];
	return max;
}