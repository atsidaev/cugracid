#pragma once

#include <stdint.h>
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

#define __int32 int32_t

class Grid
{
private:
	__int32 ReadInt32(ifstream* fs);
	double ReadDouble(ifstream* fs);
	void WriteInt32(ofstream* fs, __int32 value);
	void WriteDouble(ofstream* fs, double value);


	bool Init();

public:
	double* data;
	int nRow;
	int nCol;
	double xLL;
	double yLL;
	double xSize;
	double ySize;
	double zMin;
	double zMax;
	double Rotation;
	double BlankValue;

public:
	Grid();
	Grid(const char* fileName);

	static Grid& GenerateEmptyGrid(Grid& grid);

	double get_Average();
	double get_Min();
	double get_Max();

	bool Read(const char* fileName);
	bool Write(const char* fileName);
};

void fill_blank(Grid& grid);