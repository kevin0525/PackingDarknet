#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

extern void run_detector(int argc, char **argv);
extern void predict_main(int argc, char **argv);

int  main(int argc, char **argv)
{
	predict_main(argc,argv);
	return 0;
}