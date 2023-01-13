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
#include <sys/stat.h> 


#define ALL_TIME 0.02f
#define OUTPUT_DIR (EXP_DIR + std::string("static/output/"))
#define USE_UI
#define UI_FRAME_NUM 256
using namespace pppm;

void PPPM_test(PPPMSolver &solver, Mesh &mesh, std::vector<int3> check_cell_list, SineSource &sine, MonoPole &mp, std::string dirname)
{
    TICK(PPPMPrecompute) // 预计算开始计时
    solver.set_mesh(mesh.vertices, mesh.triangles, true); // 初始化网格和预计算网格与三角面邻居关系等,分配内存与设置缓存等

    float cost_time = TOCK_VALUE(PPPMPrecompute);
    // print cost time to "cost_time.txt"
    std::ofstream ofs0(dirname + "cost_time.txt", std::ios::out);
    ofs0 << "pppm_precompute = " << cost_time << std::endl;
    ofs0.close();

    CArr<float> neuuman_condition; // Neumann boundary(第二类边界条件)—待求变量边界外法线的方向导数被指定
    neuuman_condition.resize(solver.pg.triangles.size());
    auto triangles = solver.pg.triangles.cpu();
    int all_step = ALL_TIME / solver.dt();

    // 多点采样
    int point_num = check_cell_list.size();
    CArr2D<float> pppm_solution(point_num, all_step); // 创建储存pppm运算结果的数组，并开始计时
    TICK(PPPM)

#ifdef USE_UI
    RenderElement re(solver.pg, "PPPM"); // 如果使用UI,创建一个渲染器可视化
    re.set_params(make_int3(solver.res() / 2, 0, 0), UI_FRAME_NUM, 0.5f);
#endif

    for (int i = 0; i < all_step; i++)
    {
        solver.pg.fdtd.step(i == all_step - 1); // 进行一次时域有限差分,不加区分物体和空气地计算每个网格(包括外部边界)的当前波值
        solver.solve_fdtd_far(i == all_step - 1); // 对所有3*3*3范围内非空的网格,计算其far field(问题:far field 和 near field究竟是什么?)
        for (int p_id = 0; p_id < neuuman_condition.size(); p_id++)
        {
            auto &p = triangles[p_id];
            neuuman_condition[p_id] = (mp.neumann(p.center, p.normal) * sine(solver.dt() * i)).real(); // 指明边界法线的方向导数
        }
        solver.set_neumann_condition(neuuman_condition);
        solver.update_dirichlet(i == all_step - 1); // Dirichlet boundary这一步在做什么?
        solver.solve_fdtd_near(i == all_step - 1); // 这一步在做什么?

        // 多点采样
        for(int cnt_point = 0; cnt_point < point_num; cnt_point++)
            pppm_solution(cnt_point, i) = solver.pg.fdtd.grids[i](to_cpu(check_cell_list[cnt_point]));

#ifdef USE_UI
        if (i < UI_FRAME_NUM)
            re.assign(i, solver.pg.fdtd.grids[i]);
#endif
    }
    cost_time = TOCK_VALUE(PPPM);
    LOG("PPPM cost time: " << cost_time)

    LOG("PPPM save to " + dirname);

#ifdef USE_UI
    re.update_mesh(); // 这里生成了256*1080*1080的data，这个东西占内存并且没有被释放！
    re.write_image(UI_FRAME_NUM - 1, dirname + "/pppm.png");
    re.clear();
#endif
    write_to_txt(dirname + "pppm_solution_multi.txt", pppm_solution);
    // print cost time to "cost_time.txt"
    std::ofstream ofs2(dirname + "cost_time.txt", std::ios::app);
    ofs2 << "pppm = " << cost_time << std::endl;
    ofs2.close();
}

void Ghost_cell_test(GhostCellSolver &solver,
                     Mesh &mesh,
                     std::vector<int3> check_cell_list,
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

    int point_num = check_cell_list.size();
    CArr2D<float> ghost_cell_solution(point_num, all_step);

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

        // 多点采样
        for(int cnt_point = 0; cnt_point < point_num; cnt_point++)
            ghost_cell_solution(cnt_point, i) = solver.grid.fdtd.grids[i](to_cpu(check_cell_list[cnt_point]));
        //ghost_cell_solution[i] = solver.grid.fdtd.grids[i](to_cpu(check_cell));

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

    write_to_txt(dirname + "ghost_cell_" + order_str + "_multi.txt", ghost_cell_solution);
    // append cost time to "cost_time.txt"
    std::ofstream ofs(dirname + "cost_time.txt", std::ios::app);
    ofs << order_str + "_ghost_cell = " << cost_time << std::endl;
    ofs.close();
}

void Analytical_test(SineSource &sine, MonoPole &mp, std::vector<float3> check_point_list, int all_step, float dt, std::string dirname)
{
    CArr2D<float> analytical_solution(check_point_list.size(), all_step);
    for (int cnt_point = 0; cnt_point < check_point_list.size(); cnt_point++)
    {
        auto check_point = check_point_list[cnt_point];
        for (int i = 0; i < all_step; i++)
            analytical_solution(cnt_point, i) = (mp.dirichlet(check_point) * sine(dt * i)).real();
    }
    // for (int i = 0; i < all_step; i++)
    //     analytical_solution[i] = (mp.dirichlet(check_point) * sine(dt * i)).real();

    LOG("Analytical save to " + dirname);
    write_to_txt(dirname + "analytical_solution_multi.txt", analytical_solution);
}

int main(int argc, char *argv[])
{
    std::vector<float> grid_size_list = {0.01, 0.015, 0.02, 0.025, 0.03, 0.035};
    // std::vector<float> grid_size_list = {0.01};
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    auto obj_name = std::string("plane_thin.obj");
    auto filename = ASSET_DIR + obj_name;
    // auto mesh = Mesh::loadOBJ(filename, true);

    // auto check_point = make_float3(0.25f, 0.0f, 0.0f);
    // xcx: 多个check_point阵列
    std::vector<float3> check_point_list;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            for (int k = -2; k <= 2; k++)
                if(i*i + j*j + k*k >= 4) // 保证采样点不能离物体太近
                    check_point_list.push_back(make_float3(0.2f * i, 0.2f * j, 0.2f * k));
    // get scale from argv[0]
    float scale;
    if (argc > 1)
        scale = std::stof(argv[1]);
    else
        scale = 4.0f; // 物体的缩小比例
    for (auto grid_size : grid_size_list)
    {
        auto mesh = Mesh::loadOBJ(filename, true); // 需要在每个循环开始处重新加载mesh，否则当前mesh是上一个网格精度下fix过的mesh
        // PPPM
        float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.1); // 声波传过一个网格的时间 (为什么要除以1.1*sqrt(3)?)
        int res = 0.9 / grid_size + 2; // 总长度有多少个网格长
        PPPMSolver pppm(res, grid_size, dt); // 初始化PPPM solver，包括pg, bem, grid_far_field

        // fix mesh
        mesh.stretch_to(pppm.size().x / scale);
        mesh.move_to(pppm.center()); // 以上两行把mesh resize到适应网格大小并放到网格中心
        std::string OUT_DIR = ASSET_DIR + std::string("fixed");
        mesh.fix_mesh(grid_size, OUT_DIR, obj_name); // 这里需要对每一个grid_size各生成一个fixed_mesh

        float3 center = make_float3(res * grid_size / 2);

        // int3 check_cell = make_int3((check_point + center) / grid_size);
        std::vector<int3> check_cell_list;
        for (auto check_point : check_point_list)
            check_cell_list.push_back(make_int3((check_point + center) / grid_size));
        
        auto sine = SineSource(2 * PI * 1000); // 1000Hz正弦波源
        float wave_number = sine.omega / AIR_WAVE_SPEED; 
        auto mp = MonoPole(center, wave_number);
        std::stringstream stream1;
        stream1 << std::fixed << std::setprecision(1) << scale << "/";
        std::stringstream stream2;
        stream1 << std::fixed << std::setprecision(3) << grid_size << "/";
        auto dirname = OUTPUT_DIR + obj_name + "/" + stream1.str() + stream2.str();
        if (!std::filesystem::exists(dirname))
            std::filesystem::create_directories(dirname);
        
        // mesh.stretch_to(pppm.size().x / scale);
        mesh.move_to(pppm.center()); // 以上两行把mesh resize到适应网格大小并放到网格中心
        PPPM_test(pppm, mesh, check_cell_list, sine, mp, dirname); // 进行PPPM运算
        
        pppm.clear(); 

        // First order Ghost cell
        GhostCellSolver solver1(min_pos, grid_size, res, dt);
        solver1.set_condition_number_threshold(0.0f);
        Ghost_cell_test(solver1, mesh, check_cell_list, sine, mp, dirname, AccuracyOrder::FIRST_ORDER);
        solver1.clear(); 

        // Second order Ghost cell
        GhostCellSolver solver2(min_pos, grid_size, res, dt);
        solver2.set_condition_number_threshold(25.0f);
        Ghost_cell_test(solver2, mesh, check_cell_list, sine, mp, dirname, AccuracyOrder::SECOND_ORDER);
        solver2.clear();

        // AnalyticalSolution
        int all_step = ALL_TIME / dt;

        // 多点采样分析
        std::vector<float3> trg_pos_list;
        for (auto check_cell : check_cell_list)
            trg_pos_list.push_back(pppm.pg.getCenter(check_cell));
        // auto trg_pos = pppm.pg.getCenter(check_cell);

        Analytical_test(sine, mp, trg_pos_list, all_step, dt, dirname);
    }
}
