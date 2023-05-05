#include "../global.h"
#include "../grid/Grid.h"

int main_compare(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: compare <grid1.grd> <grid2.grd>" << endl;
		return 1;
	}
	
	Grid g1(argv[1]);
	Grid g2(argv[2]);

	if (g1.nCol != g2.nCol || g1.nRow != g2.nRow)
	{
		cout << "Error: grids should have the same size. Current sizes: " << g1.nCol << "x" << g1.nRow << ", " << g2.nCol << "x" << g2.nRow << endl;
		return 2;
	}

	bool blanks_differ = false;
	double mean = 0;
	double rms = 0;
	int count = 0;

	double g1_mean = 0;
	double g1_rms = 0;
	int g1_count = 0;

	for (auto i = 0; i < g1.nCol * g1.nRow; i++)
	{
		if ((g1.data[i] == g1.BlankValue) ^ (g2.data[i] == g2.BlankValue))
		{
			blanks_differ = true;
			continue;
		}

		if ((g1.data[i] != g1.BlankValue) && (g2.data[i] != g2.BlankValue))
		{
			count++;
			auto diff = (g1.data[i] - g2.data[i]);
			mean += abs(diff);
			rms += diff * diff;
		}

		if (g1.data[i] != g1.BlankValue)
		{
			g1_count++;
			g1_mean += abs(g1.data[i]);
			g1_rms += g1.data[i] * g1.data[i];
		}
	}

	mean /= count;
	rms = sqrt(rms / count);

	g1_mean /= g1_count;
	g1_rms = sqrt(g1_rms / g1_count);

	auto g1_amp = g1.get_Max() - g1.get_Min();

	if (blanks_differ)
		cout << "Blanked nodes position is different between grids!" << endl;

	printf("Diff: %f (%.4f%)\n", mean, (mean / g1_mean) * 100);
	printf("RMS: %f (%.4f%)\n", rms, (rms / g1_rms) * 100);
}
