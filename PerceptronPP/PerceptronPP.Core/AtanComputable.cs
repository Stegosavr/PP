
namespace PerceptronPP.Core;

public class AtanComputable : IComputable
{
	public string Name { get; } = nameof(AtanComputable);
	public double Compute(double x)
	{
		return Math.Atan(x) / Math.PI + 0.5;
	}

	public double ComputeDerivative(double x)
	{
		return 1 / (x * x + 1) / Math.PI;
	}
}