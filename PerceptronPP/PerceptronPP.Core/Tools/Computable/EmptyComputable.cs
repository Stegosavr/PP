namespace PerceptronPP.Core.Tools.Computable;

public class EmptyComputable : IComputable
{
	public string Name { get; } = nameof(EmptyComputable);
	public double Compute(double x)
	{
		return x;
	}

	public double ComputeDerivative(double x)
	{
		return 1;
	}
}