namespace PerceptronPP.Core.Tools.Weights.Provider;

public class RandomWeightsProvider : IWeightsProvider
{
	private readonly double _distribution;

	public RandomWeightsProvider(int layer, double distribution)
	{
		_distribution = distribution;
	}
	
	public double GetWeight(int _, int __)
	{
		return new Random().NextDouble() * _distribution * 2 - _distribution * 1;
	}
}