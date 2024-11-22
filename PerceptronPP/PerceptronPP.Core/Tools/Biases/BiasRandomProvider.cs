namespace PerceptronPP.Core.Tools.Biases;

public class BiasRandomProvider : IBiasProvidable
{
	private readonly double _distribution;

	public BiasRandomProvider(double distribution)
	{
		_distribution = distribution;
	}
	
	public double GetBias(int _, int __)
	{
		return (new Random()).NextDouble() * _distribution * 2 -_distribution;
	}
}