#include "gui.h"
#include "window.h"
#include "objIO.h"
#include "pppm.h"

using namespace pppm;

void test_ui()
{
	auto filename = std::string(ASSET_DIR) + std::string("/sphere.obj");
	auto mesh = loadOBJ(filename);
	mesh.normalize();
	GArr<float3> vertice = mesh.vertices;
	GArr<int3> triangles = mesh.triangles;
	GUI gui;
	CudaRender render;
	
	GArr3D<float> data(32, 32, 32);
	render.setData(data);
	render.add_mesh_to_images(vertice, triangles, make_float3(-2, -2, -2),
							  make_float3(2, 2, 2), PlaneType::XY, make_float3(0.1, 0.1, 0.1));
	gui.append(&render);
	gui.start();
}

void test_fdtd()
{
	GUI gui;
	CudaRender render;

	int res = 50;
	int step_num = 100;
	float dl = 0.005;
	float dt = 1.0f / 120000;

	GArr3D<float> data;
	data.resize(step_num, res, res);
	FDTD fdtd;
	fdtd.init(res, dl, dt);
	CArr3D<float> init_grid;
	init_grid.resize(res, res, res);
	init_grid.reset();
	init_grid(25, 25, 25) = 1;
	fdtd.grids[0].assign(init_grid);

	for (int i = 0; i < step_num; i++)
	{
		auto cpu_data = fdtd.grids[0].cpu();
		LOG(i << ": " << cpu_data(25, 25, 25));
		fdtd.copy_clip(data);
		fdtd.update();
	}

	render.setData(data, 0.01f);
	gui.append(&render);
	gui.start();
}

// argv[1] = step number
int main(int argc, char** argv)
{
	int res = 50;
	int step_num = argc > 1 ? atoi(argv[1]) : 32;
	float dl = 0.005;
	float dt = 1.0f / 120000;

	PPPMSolver solver(res, dl, dt);
	solver.precompute_p_table(step_num);
	// test_fdtd();

	return 0;
}

