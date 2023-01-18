#include <vector>
#include "array2D.h"
#include "array_writer.h"
#include "bem.h"
#include "gui.h"
#include "helper_math.h"
#include "macro.h"
#include "objIO.h"
#include "particle_grid.h"
#include "pppm.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include <filesystem>
#include <fstream>
#include "ghost_cell.h"
#include "pppm.h"
#include "RigidBody.h"
#include "progressbar.h"

using namespace pppm;
template <typename T>
void SaveGridIf(T func, const std::string &filename, int frameNum, ParticleGrid &grid, float max_value = 1.0f)
{
    if (func(frameNum))
    {
        save_grid(grid, filename, max_value, make_float3(0, 0, 0.5));
    }
}

__global__ void update_surf_acc(GArr2D<float> eigenvectors,
                                GArr<float> eigenvalues,
                                GArr<float> surface_accs,
                                int mode_idx,
                                float t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= surface_accs.size())
        return;
    float amp = eigenvectors(idx, mode_idx);
    float omega = sqrt(eigenvalues[mode_idx]);
    surface_accs[idx] = amp * sin(omega * t);
}

#define FFAT_SIZE 0.3
CGPU_FUNC int3 get_coord(int batch_idx, int x, int y, GArr3D<float> &grid)
{
    int3 e[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int3 center = make_int3(grid.size.x / 2, grid.size.y / 2, grid.size.z / 2);
    int ffat_width = grid.size.x * FFAT_SIZE;
    int i = batch_idx / 2;
    int sign = batch_idx % 2 == 0 ? -1 : 1;
    int3 p = center + e[i] * sign * ffat_width;
    int j = x - ffat_width;
    int k = y - ffat_width;
    int3 p1 = p + e[(i + 1) % 3] * j + e[(i + 2) % 3] * k;
    return p1;
}

__global__ void collect_ffat_map(GArr3D<float> result, GArr3D<float> grid)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= result.size.y || y >= result.size.z)
        return;

    int3 center = make_int3(grid.size.x / 2, grid.size.y / 2, grid.size.z / 2);
    int ffat_width = result.size.y * FFAT_SIZE / 2;
    for (int i = 0; i < 6; i++)
    {
        int3 p = get_coord(i, x, y, grid);
        result(i, x, y) += abs(grid(p));
    }
}

template <typename T>
void ffat_test(T &solver,
               RigidBody &rigidbody,
               GArr2D<float> eigenVectors,
               GArr<float> eigenValues,
               int mode_idx,
               int frame_num,
               int warm_start_frame,
               std::string dir,
               std::string tag)
{
    int ffat_size = solver.res() * FFAT_SIZE;
    GArr3D<float> result(6, ffat_size * 2, ffat_size * 2);
    result.reset();
    GArr<float> surface_accs;
    surface_accs.assign(rigidbody.surfaceAccs);
    progressbar bar(frame_num);
    float cost_time = 0;
    APPEND_TIME(cost_time, solver.set_mesh(rigidbody.tetVertices, rigidbody.tetSurfaces), SET_MESH);
    LOG("Start simulation for" << dir)
    while (bar.get_progress() <= frame_num)
    {
        cuExecute(surface_accs.size(), update_surf_acc, eigenVectors, eigenValues, surface_accs, mode_idx,
                  bar.get_progress() * solver.dt());
        APPEND_TIME(cost_time, solver.update_step(surface_accs), UPDATE_STEP);
        if (bar.get_progress() > warm_start_frame)
        {
            cuExecute2D(dim2(solver.res(), solver.res()), collect_ffat_map, result, solver.get_grid());
        }
        SaveGridIf([](int frame_num) { return frame_num == 100; }, dir + "/" + tag + "_grid.png", bar.get_progress(),
                   solver.get_particle_grid(), 1e-2);
        bar.update();
    }
    write_to_txt(dir + "/" + tag + "ffat.txt", result.data.cpu());
    write_to_txt(dir + "/" + tag + "time.txt", cost_time);
}

int main(int argc, char *argv[])
{
    std::string data_dir = DATASET_DIR + std::string("/acoustic/") + std::string(argv[1]) + "/";
    std::string OUT_DIR = data_dir + "/output/";
    CHECK_DIR(OUT_DIR);
    int res = atoi(argv[2]);
    OUT_DIR += std::to_string(res) + "/";
    CHECK_DIR(OUT_DIR);
    int mode_idx = atoi(argv[3]);
    OUT_DIR += std::to_string(mode_idx) + "/";
    CHECK_DIR(OUT_DIR);
    auto mesh = Mesh(data_dir + "mesh.obj");
    RigidBody rigidbody(data_dir, "polystyrene");
    BBox bbox = mesh.bbox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * 2.5;
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);
    rigidbody.set_sample_rate(frame_rate);
    rigidbody.fix_mesh(grid_size, OUT_DIR);
    float dt = 1.0f / frame_rate;
    LOG("min pos: " << min_pos);
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    GArr2D<float> eigenVectors = rigidbody.modelMatrixSurf;
    LOG("eigenVectors: (" << eigenVectors.size.x << ", " << eigenVectors.size.y << ")")
    GArr<float> eigenValues = rigidbody.eigenVals;
    LOG("eigenValues: (" << eigenValues.size() << ")")
    auto eigenVecsCpu = eigenVectors.cpu();
    auto eigenValsCpu = eigenValues.cpu();

    PPPMSolver solver(res, grid_size, dt, min_pos);

    float max_time = 0.02f;
    float warm_start_time = 0.01f;
    int warm_start_frame = warm_start_time / dt;
    int frame_num = max_time / dt;

    int ffat_size = solver.res() * FFAT_SIZE;
    CArr3D<float3> pixel_pos(6, ffat_size * 2, ffat_size * 2);
    auto grid = solver.get_grid();
    for (int i = 0; i < 6; i++)
        for (int x = 0; x < ffat_size * 2; x++)
            for (int y = 0; y < ffat_size * 2; y++)
            {
                int3 pos = get_coord(i, x, y, grid);
                pixel_pos(i, x, y) = solver.pg.getCenter(pos.x, pos.y, pos.z);
            }
    write_to_txt(OUT_DIR + "/pixel_pos.txt", pixel_pos.data);
    ffat_test(solver, rigidbody, eigenVectors, eigenValues, mode_idx, frame_num, warm_start_frame, OUT_DIR, "pppm_");
    solver.clear();

    GhostCellSolver solver2(min_pos, grid_size, res, dt);
    solver2.set_condition_number_threshold(0);
    ffat_test(solver2, rigidbody, eigenVectors, eigenValues, mode_idx, frame_num, warm_start_frame, OUT_DIR, "ghost1_");
    solver2.clear();
    GhostCellSolver solver3(min_pos, grid_size, res, dt);
    solver3.set_condition_number_threshold(25);
    ffat_test(solver3, rigidbody, eigenVectors, eigenValues, mode_idx, frame_num, warm_start_frame, OUT_DIR, "ghost2_");
    solver3.clear();
    CArr<float> omega;
    CArr<float> surf_neumann;
    omega.resize(1);
    surf_neumann.resize(rigidbody.surfaceAccs.size());
    omega[0] = std::sqrt(eigenValsCpu[mode_idx]);
    for (int i = 0; i < rigidbody.surfaceAccs.size(); i++)
        surf_neumann[i] = eigenVecsCpu(i, mode_idx);
    write_to_txt(OUT_DIR + "/surf_neumann.txt", surf_neumann);
    write_to_txt(OUT_DIR + "/omega.txt", omega);
    rigidbody.clear();
    solver.clear();
    solver2.clear();
    return 0;
}
