#include <string>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
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
#include <sys/stat.h>

using namespace pppm;

#define ALL_TIME 0.01
#define SKIP_FRAME 500
#define CHECK_BBOX_NUM 4
#define CHECK_BBOX_SIZE        \
    {                          \
        0.25, 0.30, 0.35, 0.40 \
    }

__global__ void update_surf_acc(SineSource sine,
                                MonoPole mp,
                                GArr<float> surface_accs,
                                GArr<Triangle> triangles,
                                float t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= surface_accs.size())
        return;
    auto &p = triangles[idx];
    surface_accs[idx] = (mp.neumann(p.center, p.normal) * sine(t)).real();
}

__global__ void collect_ffat_map(GArr3D<float> result, GArr3D<float> grid)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= result.size.y || y >= result.size.z)
        return;
    int3 e[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int3 center = make_int3(grid.size.x / 2, grid.size.y / 2, grid.size.z / 2);
    int ffat_width = result.size.y / 2;
    int batch_idx = -1;
    for (int i = 0; i < 3; i++)
        for (int sign = -1; sign <= 1; sign += 2)
        {
            batch_idx++;
            int3 p = center + e[i] * sign * ffat_width;
            for (int j = -ffat_width; j < ffat_width; j++)
                for (int k = -ffat_width; k < ffat_width; k++)
                {
                    int3 p1 = p + e[(i + 1) % 3] * j + e[(i + 2) % 3] * k;
                    result(batch_idx, x, y) += abs(grid(p1));
                }
        }
}

template <typename T>
void points_test(T &solver, Mesh &mesh, SineSource &sine, MonoPole &mp, std::string dirname)
{
    GArr<float> surface_accs;
    auto triangles = solver.get_triangles();
    surface_accs.resize(mesh.triangles.size());
    int all_step = ALL_TIME / solver.dt();
    int start_clip = 500;
    CHECK_DIR(dirname);
    float cost_time = 0;

    APPEND_TIME(cost_time, solver.set_mesh(mesh.vertices, mesh.triangles), SET_MESH)
    GArr3D<float> check_ffat[CHECK_BBOX_NUM];
    float check_ffat_size[CHECK_BBOX_NUM] = CHECK_BBOX_SIZE;
    for (int i = 0; i < CHECK_BBOX_NUM; i++)
    {
        int ffat_width = check_ffat_size[i] * solver.res();
        ffat_width *= 2;
        check_ffat[i].resize(6, ffat_width, ffat_width);
    }

    for (int i = 0; i < all_step; i++)
    {
        cuExecute(surface_accs.size(), update_surf_acc, sine, mp, surface_accs, triangles, solver.dt() * i);
        APPEND_TIME(cost_time, solver.update_step(surface_accs), UPDATE_STEP)
        if (i > SKIP_FRAME)
            for (int j = 0; j < 6; j++)
            {
                cuExecute2D(dim2(check_ffat[j].rows, check_ffat[j].cols), collect_ffat_map, check_ffat[0],
                            solver.get_grid());
            }
    }
    for (int i = 0; i < 6; i++)
        write_to_txt(dirname + "/ffat" + std::to_string(i) + ".txt", check_ffat[i].data.cpu());
    write_to_txt(dirname + "/cost_time.txt", cost_time);
    surface_accs.clear();
    check_ffat->clear();
}

class GroudTruth
{
    public:
        float grid_size;
        float3 min_pos;
        int grid_dim;
        GArr3D<float> grid;
        int t;
        float delta_t;
        SineSource sine;
        MonoPole mp;
        GArr<Triangle> triangles;
        GroudTruth(float3 min_pos_,
                   int res_,
                   float grid_size_,
                   float dt_,
                   SineSource sine_,
                   MonoPole mp_,
                   GArr<Triangle> triangles_)
        {
            min_pos = min_pos_;
            grid_dim = res_;
            grid_size = grid_size_;
            grid.resize(grid_dim, grid_dim, grid_dim);
            t = 0;
            delta_t = dt_;
            sine = sine_;
            mp = mp_;
            triangles = triangles_;
        }
        void set_mesh(CArr<float3> vertices, CArr<int3> triangles){};
        CGPU_FUNC inline float dt() { return delta_t; }
        CGPU_FUNC inline int res() { return grid_dim; }
        GArr3D<float> get_grid() { return grid; }
        CGPU_FUNC inline float3 getCenter(int i, int j, int k) const
        {
            return make_float3((i + 0.5f) * grid_size, (j + 0.5f) * grid_size, (k + 0.5f) * grid_size) + min_pos;
        }
        void update_step(GArr<float> surface_accs);
        void clear() { grid.clear(); }
        GArr<Triangle> get_triangles() { return triangles; }
};

__global__ void groundtruth_update(SineSource sine, MonoPole mp, GroudTruth gt, float t)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= gt.res() || y >= gt.res() || z >= gt.res())
        return;
    float3 p = gt.getCenter(x, y, z);
    gt.grid(x, y, z) = (mp.dirichlet(p) * sine(0)).real();
}

void GroudTruth::update_step(GArr<float> surface_accs)
{
    cuExecute3D(dim3(res(), res(), res()), groundtruth_update, sine, mp, *this, t * delta_t);
    t++;
}

int main(int argc, char *argv[])
{
    std::vector<float> grid_size_list = {0.01, 0.015, 0.02, 0.025, 0.03, 0.035};
    auto dir_name = ROOT_DIR + std::string(argv[1]) + "/";
    auto OUT_DIR = dir_name + "output/";
    CHECK_DIR(OUT_DIR);

    float scale = 6.0;
    float box_size = 0.7;
    for (auto grid_size : grid_size_list)
    {
        auto mesh = Mesh::loadOBJ(dir_name + "mesh.obj");
        mesh.stretch_to(box_size / scale);
        mesh.fix_mesh(grid_size, OUT_DIR);
        auto OUT_SUB_DIR = OUT_DIR + std::to_string(grid_size) + "/";
        CHECK_DIR(OUT_SUB_DIR);
        BBox bbox = mesh.bbox();
        float3 min_pos = bbox.min;
        float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.01);
        int res = box_size / grid_size;
        GArr<Triangle> triangles;

        auto sine = SineSource(2 * PI * 1000);
        float wave_number = sine.omega / AIR_WAVE_SPEED;
        auto mp = MonoPole(mesh.get_center(), wave_number);

        // PPPM
        PPPMSolver pppm(res, grid_size, dt, min_pos);
        points_test(pppm, mesh, sine, mp, OUT_SUB_DIR + "/pppm/");
        triangles.assign(pppm.pg.triangles);
        pppm.clear();

        // First order Ghost cell
        GhostCellSolver solver1(min_pos, grid_size, res, dt);
        solver1.set_condition_number_threshold(0.0f);
        points_test(solver1, mesh, sine, mp, OUT_SUB_DIR + "/ghostcell1/");
        solver1.clear();

        // Second order Ghost cell
        GhostCellSolver solver2(min_pos, grid_size, res, dt);
        solver2.set_condition_number_threshold(25.0f);
        points_test(solver2, mesh, sine, mp, OUT_SUB_DIR + "/ghostcell2/");
        solver2.clear();

        // Ground truth
        GroudTruth gt(min_pos, res, grid_size, dt, sine, mp, triangles);
        points_test(gt, mesh, sine, mp, OUT_SUB_DIR + "/groundtruth/");
        gt.clear();
    }
}
