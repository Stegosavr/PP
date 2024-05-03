namespace PerceptronPP.Core.Tools.Weights;

public class RandomWeightsAccessor : IWeightsAccessible
{
	private readonly double _distribution;

	public RandomWeightsAccessor(int layer)
	{
		_distribution = (1 / Math.Sqrt(layer));
	}
	
	public double GetWeight(int _, int __)
	{
		return new Random().NextDouble() * _distribution * 2 - _distribution;
	}
}