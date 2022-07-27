#include "gui.h"
#include "particle_grid.h"
#include "window.h"
#include "objIO.h"
using namespace pppm;

int main()
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
	return 0;
}
