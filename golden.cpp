#include <math.h>
typedef double (*function)(double i);

double golden_section(function f, int min, int max, int steps_count)
{
	double a = min, b = max;

	double g = a+(b-a)*(3-sqrt((double)5))/2;
	double h = a+(b-a)*(sqrt((double)5)-1)/2;

	double fu1 = f(g);
	double fu2 = f(h);

	for (int i = 1; i <= steps_count; i++)
	{
		if( fu1<=fu2 )
		{
			b = h;
			h = g;
			fu2 = fu1;
			g = a+(b-a)*(3-sqrt((double)5))/2;
			fu1 = f(g);
		}
		else
		{
			a = g;
			g = h;
			fu1 = fu2;
			h = a+(b-a)*(sqrt((double)5)-1)/2;
			fu2 = f(h);
		}
	}
	return (fu1-fu2)/2+fu2;
}
