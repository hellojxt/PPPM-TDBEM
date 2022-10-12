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

class RenderElement
{
    public:
        GArr3D<float> data;
        float max_abs_value;
        std::string name;
        RenderElement(GArr3D<float> data_, float max_abs_value_, const char *name_)
            : data(data_), max_abs_value(max_abs_value_), name(name_)
        {}
};

static inline void renderArray(RenderElement e)
{
    GUI gui;
    CudaRender render(e.name.c_str());
    render.setData(e.data, e.max_abs_value);
    gui.append(&render);
    gui.start();
}

static inline void renderArray(RenderElement e1, RenderElement e2)
{
    GUI gui;
    CudaRender render1(e1.name.c_str());
    render1.setData(e1.data, e1.max_abs_value);
    gui.append(&render1);
    CudaRender render2(e2.name.c_str());
    render2.setData(e2.data, e2.max_abs_value);
    gui.append(&render2);
    gui.start();
}

}  // namespace pppm