#pragma once
#include <stdio.h>
#include "window.h"
#include "array.h"

namespace pppm
{
    class GUI
    {
        CArr<Window *> sub_windows;

    public:
        void start(int height = 1500, int width = 900);
        void update();
        void append(Window *window);
    };

}