#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "objIO.h"
#include "pppm.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include <filesystem>
#include <fstream>
#include "ghost_cell.h"

#define ALL_TIME 0.02f
#define OUTPUT_DIR (EXP_DIR + std::string("static/output/"))
#define USE_UI
#define UI_FRAME_NUM 256
using namespace pppm;

void PPPM_test(PPPMSolver &solver, Mesh &mesh, int3 check_cell, SineSource &sine, MonoPole &mp, std::string dirname)
{
    TICK(PPPMPrecompute)
    solver.set_mesh(mesh.vertices, mesh.triangles, true);

    float cost_time = TOCK_VALUE(PPPMPrecompute);
    // print cost time to "cost_time.txt"
    std::ofstream ofs0(dirname + "cost_time.txt", std::ios::out);
    ofs0 << "pppm_precompute = " << cost_time << std::endl;
    ofs0.close();

    CArr<float> neuuman_condition;
    neuuman_condition.resize(solver.pg.triangles.size());
    auto triangles = solver.pg.triangles.cpu();

    int all_step = ALL_TIME / solver.dt();
    CArr<float> pppm_solution(all_step);
    TICK(PPPM)

#ifdef USE_UI
    RenderElement re(solver.pg, "PPPM");
    re.set_params(make_int3(solver.res() / 2, 0, 0), UI_FRAME_NUM, 0.5f);
#endif

    for (int i = 0; i < all_step; i++)
    {
        solver.pg.fdtd.step(i == all_step - 1);
        solver.solve_fdtd_far(i == all_step - 1);
        for (int p_id = 0; p_id < neuuman_condition.size(); p_id++)
        {
            auto &p = triangles[p_id];
            neuuman_condition[p_id] = (mp.neumann(p.center, p.normal) * sine(solver.dt() * i)).real();
        }
        solver.set_neumann_condition(neuuman_condition);
        solver.update_dirichlet(i == all_step - 1);
        solver.solve_fdtd_near(i == all_step - 1);
        pppm_solution[i] = solver.pg.fdtd.grids[i](to_cpu(check_cell));
#ifdef USE_UI
        if (i < UI_FRAME_NUM)
            re.assign(i, solver.pg.fdtd.grids[i]);
#endif
    }
    cost_time = TOCK_VALUE(PPPM);
    LOG("PPPM cost time: " << cost_time)

    LOG("PPPM save to " + dirname);

#ifdef USE_UI
    re.update_mesh();
    re.write_image(UI_FRAME_NUM - 1, dirname + "/pppm.png");
    re.clear();
#endif
    write_to_txt(dirname + "pppm_solution.txt", pppm_solution);
    // print cost time to "cost_time.txt"
    std::ofstream ofs2(dirname + "cost_time.txt", std::ios::app);
    ofs2 << "pppm = " << cost_time << std::endl;
    ofs2.close();
}

void Ghost_cell_test(GhostCellSolver &solver,
                     Mesh &mesh,
                     int3 check_cell,
                     SineSource &sine,
                     MonoPole &mp,
                     std::string dirname,
                     AccuracyOrder order)
{
    TICK(GhostCell)
    solver.set_mesh(mesh.vertices, mesh.triangles);
    CArr<float> neuuman_condition;
    neuuman_condition.resize(solver.grid.triangles.size());
    auto triangles = solver.grid.triangles.cpu();

    solver.precompute_cell_data(true);
    solver.precompute_ghost_matrix(true);
    int all_step = ALL_TIME / solver.dt();
    CArr<float> ghost_cell_solution(all_step);

#ifdef USE_UI
    RenderElement re(solver.grid, "GhostCell");
    re.set_params(make_int3(solver.res() / 2, 0, 0), UI_FRAME_NUM, 0.5f);
#endif

    for (int i = 0; i < all_step; i++)
    {
        solver.grid.fdtd.step(i == all_step - 1);
        for (int p_id = 0; p_id < neuuman_condition.size(); p_id++)
        {
            auto &p = triangles[p_id];
            neuuman_condition[p_id] = (mp.neumann(p.center, p.normal) * sine(solver.dt() * i)).real();
        }
        solver.set_boundary_condition(neuuman_condition);
        solver.solve_ghost_cell(i == all_step - 1);
        ghost_cell_solution[i] = solver.grid.fdtd.grids[i](to_cpu(check_cell));
#ifdef USE_UI
        if (i < UI_FRAME_NUM)
            re.assign(i, solver.grid.fdtd.grids[i]);
#endif
    }
    float cost_time = TOCK_VALUE(GhostCell);
    LOG("GhostCell cost time: " << cost_time)

    LOG(order << " ghost cell save to " + dirname);
    std::string order_str = (order == AccuracyOrder::FIRST_ORDER) ? "1st" : "2nd";
#ifdef USE_UI
    re.update_mesh();
    re.write_image(UI_FRAME_NUM - 1, dirname + "/ghost_cell_" + order_str + ".png");
    re.clear();
#endif

    write_to_txt(dirname + "ghost_cell_" + order_str + ".txt", ghost_cell_solution);
    // append cost time to "cost_time.txt"
    std::ofstream ofs(dirname + "cost_time.txt", std::ios::app);
    ofs << order_str + "_ghost_cell = " << cost_time << std::endl;
    ofs.close();
}

void Analytical_test(SineSource &sine, MonoPole &mp, float3 check_point, int all_step, float dt, std::string dirname)
{
    CArr<float> analytical_solution(all_step);
    for (int i = 0; i < all_step; i++)
        analytical_solution[i] = (mp.dirichlet(check_point) * sine(dt * i)).real();
    LOG("Analytical save to " + dirname);
    write_to_txt(dirname + "analytical_solution.txt", analytical_solution);
}

int main(int argc, char *argv[])
{
    std::vector<float> grid_size_list = {0.01, 0.015, 0.02, 0.025, 0.03, 0.035};
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    auto obj_name = std::string("sphere4.obj");
    auto filename = ASSET_DIR + obj_name;
    auto mesh = Mesh::loadOBJ(filename, true);
    auto check_point = make_float3(0.25f, 0.0f, 0.0f);
    // get scale from argv[0]
    float scale;
    if (argc > 1)
        scale = std::stof(argv[1]);
    else
        scale = 4.0f;
    for (auto grid_size : grid_size_list)
    {
        float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.1);
        int res = 0.7 / grid_size + 2;
        float3 center = make_float3(res * grid_size / 2);
        int3 check_cell = make_int3((check_point + center) / grid_size);
        auto sine = SineSource(2 * PI * 1000);
        float wave_number = sine.omega / AIR_WAVE_SPEED;
        auto mp = MonoPole(center, wave_number);
        std::stringstream stream1;
        stream1 << std::fixed << std::setprecision(1) << scale << "/";
        std::stringstream stream2;
        stream1 << std::fixed << std::setprecision(3) << grid_size << "/";
        auto dirname = OUTPUT_DIR + obj_name + "/" + stream1.str() + stream2.str();
        if (!std::filesystem::exists(dirname))
            std::filesystem::create_directories(dirname);
        // PPPM
        PPPMSolver pppm(res, grid_size, dt);
        mesh.stretch_to(pppm.size().x / scale);
        mesh.move_to(pppm.center());
        PPPM_test(pppm, mesh, check_cell, sine, mp, dirname);
        pppm.clear();

        // First order Ghost cell
        GhostCellSolver solver1(min_pos, grid_size, res, dt);
        solver1.set_condition_number_threshold(0.0f);
        Ghost_cell_test(solver1, mesh, check_cell, sine, mp, dirname, AccuracyOrder::FIRST_ORDER);
        solver1.clear();

        // Second order Ghost cell
        GhostCellSolver solver2(min_pos, grid_size, res, dt);
        solver2.set_condition_number_threshold(25.0f);
        Ghost_cell_test(solver2, mesh, check_cell, sine, mp, dirname, AccuracyOrder::SECOND_ORDER);
        solver2.clear();

        // AnalyticalSolution
        int all_step = ALL_TIME / dt;
        auto trg_pos = pppm.pg.getCenter(check_cell);
        Analytical_test(sine, mp, trg_pos, all_step, dt, dirname);
    }
}
