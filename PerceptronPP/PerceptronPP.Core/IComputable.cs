namespace PerceptronPP.Core;

public interface IComputable
{
	public string Name { get; }
	public double Compute(double x);
	public double ComputeDerivative(double x);
}