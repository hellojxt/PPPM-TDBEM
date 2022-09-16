#pragma once
#include "array.h"
#include <cstring>
#include <stdio.h>
#include <stdlib.h>

namespace pppm
{
    void write_to_txt(const char *filename, cpx *data, int num)
    {
        FILE *fp = fopen(filename, "w");
        for (int i = 0; i < num; i++)
        {
            fprintf(fp, "%e %e\n", data[i].real(), data[i].imag());
        }
        fclose(fp);
    }

}
