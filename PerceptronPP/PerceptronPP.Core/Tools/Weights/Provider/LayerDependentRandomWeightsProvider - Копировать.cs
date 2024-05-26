namespace PerceptronPP.Core.Tools.Weights.Provider;

public class LayerDependentWeightsProvider : IWeightsProvider
{
	private readonly double _distribution;

	public LayerDependentWeightsProvider(int layer, double distribution)
	{
		_distribution = (1 / Math.Sqrt(layer)) * distribution;
	}
	
	public double GetWeight(int _, int __)
	{
		return new Random().NextDouble() * _distribution * 2 - _distribution * 1;
	}
}