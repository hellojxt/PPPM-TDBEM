#pragma once
#include <stdio.h>
#include "array.h"
#include "window.h"

namespace pppm
{
class GUI
{
    CArr<Window *> sub_windows;

  public:
    void start(int height = 1600, int width = 900);
    void update();
    void append(Window *window);
};

}  // namespace pppm