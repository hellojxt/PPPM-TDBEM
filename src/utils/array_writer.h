#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "array3D.h"
#include "png.h"

namespace pppm
{

void static inline write_to_txt(std::string filename, cpx *data, int num)
{
    FILE *fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < num - 1; i++)
    {
        fprintf(fp, "%e %e\n", data[i].real(), data[i].imag());
    }
    fprintf(fp, "%e %e", data[num - 1].real(), data[num - 1].imag());
    fclose(fp);
}

void static inline write_to_txt(std::string filename, float *data, int num)
{
    FILE *fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < num - 1; i++)
    {
        fprintf(fp, "%e\n", data[i]);
    }
    fprintf(fp, "%e", data[num - 1]);
    fclose(fp);
}

void static inline write_to_txt(std::string filename, float3 *data, int num)
{
    FILE *fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < num - 1; i++)
    {
        fprintf(fp, "%e %e %e\n", data[i].x, data[i].y, data[i].z);
    }
    fprintf(fp, "%e %e %e", data[num - 1].x, data[num - 1].y, data[num - 1].z);
    fclose(fp);
}

void static inline write_to_txt(std::string filename, uchar4 *data, int num)
{
    FILE *fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < num - 1; i++)
    {
        fprintf(fp, "%d %d %d %d\n", data[i].x, data[i].y, data[i].z, data[i].w);
    }
    fprintf(fp, "%d %d %d %d", data[num - 1].x, data[num - 1].y, data[num - 1].z, data[num - 1].w);
    fclose(fp);
}

template <typename T>
void static inline write_to_txt(std::string filename, CArr<T> data)
{
    write_to_txt(filename, data.data(), data.size());
}

void static inline write_to_txt(std::string filename, float data)
{
    CArr<float> arr(1);
    arr[0] = data;
    write_to_txt(filename, arr);
}

void static inline write_to_png(std::string filename, uchar4 *data, int width, int height)
{
    FILE *fp = fopen(filename.c_str(), "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    png_bytep row = (png_bytep)malloc(4 * width * sizeof(png_byte));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            row[j * 4 + 0] = data[i * width + j].x;
            row[j * 4 + 1] = data[i * width + j].y;
            row[j * 4 + 2] = data[i * width + j].z;
            row[j * 4 + 3] = data[i * width + j].w;
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);
    fclose(fp);
    free(row);
}

void static inline write_to_png(std::string filename, CArr2D<uchar4> data)
{
    CArr2D<uchar4> img(data.cols, data.rows);
    for (int i = 0; i < data.rows; i++)
    {
        for (int j = 0; j < data.cols; j++)
        {
            img(data.cols - j - 1, i) = data(i, j);
        }
    }
    write_to_png(filename, img.begin(), img.rows, img.cols);
}

CArr<float> static inline read_from_txt(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    fseek(fp, 0, SEEK_END);
    int num = ftell(fp) / sizeof(float);
    fseek(fp, 0, SEEK_SET);
    CArr<float> data(num);
    for (int i = 0; i < num; i++)
    {
        fscanf(fp, "%e", &data[i]);
    }
    fclose(fp);
    return data;
}

}  // namespace pppm
