using MathNet.Numerics.LinearAlgebra;
namespace PerceptronPP.Core
{
    public class GradientDescentData
    {
        public readonly Matrix<double> PreviousWeightsGradient;
        public readonly Matrix<double> PreviousBiasesGradient;

        public GradientDescentData(Matrix<double> prevWeightsGrad, Matrix<double> prevBiasesGrad)
        {
            PreviousWeightsGradient = prevWeightsGrad;
            PreviousBiasesGradient = prevBiasesGrad;
        }
    }
}