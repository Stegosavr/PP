namespace PerceptronPP.Core.Tools.Computable;

public class SigmoidComputable : IComputable
{
	public string Name { get; } = nameof(SigmoidComputable);
	public double Compute(double x)
	{
		return 1 / (1 + Math.Exp(-x));
	}

	public double ComputeDerivative(double x)
	{
		var sigmoid = Compute(x);
        return sigmoid * (1 - sigmoid);
	}
}