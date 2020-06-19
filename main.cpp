#include <iostream>
#include <string.h>

#include "global.h"

typedef int main_ptr(int, char**);

typedef struct geo_prog_t {
	char* name;
	main_ptr* main;
} geo_prog_t;

geo_prog_t progs[] {
	{ "v3", main_v3 },
	{ "lc", main_lc },
	{ "m2km", main_m2km },
};

int main(int argc, char** argv)
{
	geo_prog_t* program = nullptr;
	bool shift_args = false;

	for (int i = 0; i < sizeof(progs) / sizeof(geo_prog_t); i++)
	{
		if (strcmp(progs[i].name, argv[0]) == 0)
		{
			program = &progs[i];
			break;
		}
	}

	if (program == nullptr && argc > 1)
	{
		for (int i = 0; i < sizeof(progs) / sizeof(geo_prog_t); i++)
		{
			if (strcmp(progs[i].name, argv[1]) == 0)
			{
				program = &progs[i];
				shift_args = true;
				break;
			}
		}
	}

	if (program == nullptr)
	{
		std::cout << "Please create symbolic link with PROGRAM name to this executable. Also you may execute as " << argv[0] << " PROGRAM [args...]." << std::endl;
		std::cout << "PROGRAM may be: ";
		for (int i = 0; i < sizeof(progs) / sizeof(geo_prog_t); i++)
			std::cout << progs[i].name << " ";
		std::cout << std::endl;
	}
	else
	{
		if (shift_args)
		{
			argc--;
			argv = &argv[1];
		}

		program->main(argc, argv);
	}
}
