#pragma once
#include <string>
namespace pppm
{

class MaterialParameters
{

    public:
        float youngModulus;
        float poissonRatio;
        float density;
        float alpha;
        float beta;
        float gamma;
        float restitution_coeff;
        float friction_coeff;

        MaterialParameters()
        {
            youngModulus = 0.0f;
            poissonRatio = 0.0f;
            density = 0.0f;
            alpha = 0.0f;
            beta = 0.0f;
            gamma = 0.0f;
            restitution_coeff = 0.0f;
            friction_coeff = 0.0f;
        }

        void set_parameters(const std::string &material_name)
        {
            if (material_name == "ceramic")
            {
                youngModulus = 7.4e10;
                poissonRatio = 0.19;
                density = 2300.0;
                alpha = 40.0;
                beta = 1e-7;
                gamma = 3e-2;
                restitution_coeff = 0.4;
                friction_coeff = 0.2;
            }
            else if (material_name == "polystyrene")
            {
                youngModulus = 3.5e9;
                poissonRatio = 0.34;
                density = 1050.0;
                alpha = 30.0;
                beta = 8e-7;
                gamma = 4e-4;
                restitution_coeff = 0.4;
                friction_coeff = 0.2;
            }
            else if (material_name == "steel")
            {
                youngModulus = 2e11;
                poissonRatio = 0.29;
                density = 7850.0;
                alpha = 5.0;
                beta = 3e-8;
                gamma = 9e-3;
                restitution_coeff = 0.8;
                friction_coeff = 0.3;
            }
            else if (material_name == "mdf")
            {
                youngModulus = 4e9;
                poissonRatio = 0.32;
                density = 615.0;
                alpha = 35.0;
                beta = 5e-6;
                gamma = 9e-3;
                restitution_coeff = 0.4;
                friction_coeff = 0.2;
            }
            else if (material_name == "wood")
            {
                youngModulus = 1.1e10;
                poissonRatio = 0.25;
                density = 750.0;
                alpha = 60.0;
                beta = 2e-6;
                gamma = 5e-4;
                restitution_coeff = 0.4;
                friction_coeff = 0.2;
            }
            else
            {
                youngModulus = 0.0f;
                poissonRatio = 0.0f;
                density = 0.0f;
                alpha = 0.0f;
                beta = 0.0f;
                gamma = 0.0f;
                restitution_coeff = 0.0f;
                friction_coeff = 0.0f;
            }
        };
};

}  // namespace pppm