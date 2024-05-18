using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.GradientDescent;

namespace PerceptronPP.Core.Tools.GradientDescent.Optimizers;

public class StochasticGradientDescent : IOptimizer
{
    public string Name { get; } = nameof(StochasticGradientDescent);
    public void GradientDescent(ParameterType _, ref Matrix<double> currentParameters, 
        Matrix<double> parametersDerivative, double coefficient, int __)
    {
        currentParameters -= parametersDerivative * coefficient;
    }
}
