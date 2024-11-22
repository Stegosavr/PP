using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP.Core.Tools.GradientDescent;

public class GradientDescentData
{
    public Matrix<double> WeightsDerivative;
    public Matrix<double> BiasesDerivative;
    public Matrix<double> PreviousWeightsVelocity;
    public Matrix<double> PreviousBiasesVelocity;

    public GradientDescentData()
    {
    }

    public void SetPreviousVelocity(ParameterType type, Matrix<double> value)
    {
        if (type is ParameterType.Weight)
            PreviousWeightsVelocity = value.Clone();
        if (type == ParameterType.Bias)
            PreviousBiasesVelocity = value.Clone();
    }

    public Matrix<double> GetPreviousVelocity(ParameterType type)
    {
        if (type is ParameterType.Weight)
            return PreviousWeightsVelocity;
        if (type == ParameterType.Bias)
            return PreviousBiasesVelocity;
        throw new ArgumentException();
    }
}
