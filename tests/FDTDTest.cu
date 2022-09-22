#include "gui.h"
#include "window.h"
#include "objIO.h"
#include "fdtd.h"
#include "sound_source.h"

using namespace pppm;

__global__ void set_center_signal(FDTD fdtd, SineSource s){
    int3 center = make_int3(fdtd.res / 2, fdtd.res / 2, fdtd.res / 2);
    fdtd.grids[fdtd.t - 1](center) = s((fdtd.t - 1) * fdtd.dt).real(); // ftdt.t is the next time step
}

int main()
{
	GUI gui;
	CudaRender render;

    int res = 51;
	int step_num = 300;
	float dl = 0.005;
	float dt = 1.0f / 150000;

	GArr3D<float> data;
    data.resize(step_num, res, res);
	FDTD fdtd;
	fdtd.init(res, dl, dt);
    SineSource s(5000 * 2 * M_PI);

	for (int i = 0; i < step_num; i++)
	{
		fdtd.step();
        cuExecuteBlock(1, 1, set_center_signal, fdtd, s);
		data[i].assign(fdtd.grids[i][25]);
	}
	render.setData(data, 0.02f);
	gui.append(&render);
	gui.start();
}