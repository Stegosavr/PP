using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.GradientDescent;

namespace PerceptronPP.Core.Tools.GradientDescent.Optimizers;

public interface IOptimizer
{
	public string Name { get; }
	public void GradientDescent(ParameterType type,ref Matrix<double> currentParameters,
		Matrix<double> parametersDerivative, double coefficient, int layerIndex);
}