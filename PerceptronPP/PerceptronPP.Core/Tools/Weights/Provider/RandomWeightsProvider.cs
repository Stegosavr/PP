namespace PerceptronPP.Core.Tools.Weights.Provider;

public class RandomWeightsProvider : IWeightsProvider
{
	private readonly double _distribution;

	public RandomWeightsProvider(int layer)
	{
		_distribution = (1 / Math.Sqrt(layer));
	}
	
	public double GetWeight(int _, int __)
	{
		return new Random().NextDouble() * _distribution * 50 - _distribution * 25;
	}
}