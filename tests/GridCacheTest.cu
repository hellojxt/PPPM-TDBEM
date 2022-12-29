#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "pppm.h"
#include "sound_source.h"
#include "window.h"

using Catch::Approx;
// !!!!!!!!!!!!!!!!!!----- History size must be 2*STEP_NUM for this test -----!!!!!!!!!!!!!!!!!!!!!
TEST_CASE("GridCache", "[gc]")
{
    using namespace pppm;
    PPPMSolver *solver = regular_random_pppm(256);

    auto dirichlet = solver->dirichlet.cpu();
    for (int t_idx = 0; t_idx < STEP_NUM; t_idx++)
    {
        for (int i = 0; i < dirichlet.size(); i++)
        {
            dirichlet[i][-t_idx - 1] = RAND_I(1, 5);
        }
    }

    solver->dirichlet.assign(dirichlet);

    solver->solve_fdtd_far_simple();
    solver->solve_fdtd_near_simple();
    CArr3D<float> far_field_simple = solver->grid_far_field[0].cpu();
    CArr3D<float> fdtd_grid_simple = solver->pg.fdtd.grids[0].cpu();

    solver->pg.fdtd.reset();
    solver->pg.fdtd.step();
    solver->solve_fdtd_far();
    solver->solve_fdtd_near();
    CArr3D<float> far_field = solver->grid_far_field[0].cpu();
    CArr3D<float> fdtd_grid = solver->pg.fdtd.grids[0].cpu();
    auto grid_list = solver->pg.grid_face_list.cpu();
    int assert_num = 0;
    for (int x = 1; x < solver->res() - 1; x++)
    {
        for (int y = 1; y < solver->res() - 1; y++)
        {
            for (int z = 1; z < solver->res() - 1; z++)
            {
                if (far_field_simple(x, y, z) != 0)
                {
                    REQUIRE(far_field_simple(x, y, z) == Approx(far_field(x, y, z)).margin(1e-3));
                    assert_num++;
                }
                if (fdtd_grid_simple(x, y, z) != 0)
                {
                    REQUIRE(fdtd_grid_simple(x, y, z) == Approx(fdtd_grid(x, y, z)).margin(1e-3));
                    assert_num++;
                }
            }
        }
    }
}
