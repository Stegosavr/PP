using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.GradientDescent;

namespace PerceptronPP.Core.Tools.GradientDescent.Optimizers;

public class sample : IOptimizer
{
    public string Name { get; } = nameof(StochasticGradientDescent);
    private GradientDescentData _gradientDescentData;
    public void GradientDescent(ref Matrix<double> currentParameters, 
        Matrix<double> parametersDerivative, double coefficient, int _)
    {
        currentParameters -= parametersDerivative * coefficient;
    }

    public void GradientDescent(ParameterType type, ref Matrix<double> currentParameters, Matrix<double> parametersDerivative, double coefficient, int layerIndex)
    {
        throw new NotImplementedException();
    }
}

