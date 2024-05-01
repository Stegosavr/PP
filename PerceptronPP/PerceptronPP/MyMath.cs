using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP
{
    static class MyMath
    {
        public static double Arctan(double x)
        {
            return Math.Atan(x) / Math.PI + 0.5;
        }

        public static double ArctanDerivative(double x)
        {
            return 1 / (x * x + 1) / Math.PI;
        }
    }
}
