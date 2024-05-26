using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core.Tools.Weights.Factory;

public class RandomWeightsFactory : IWeightsFactory
{
	private readonly Func<int, int> _getCount;
    private readonly WeightsProviderType _providerType;
    private readonly double _distribution;

    public RandomWeightsFactory(Func<int, int> getCount, WeightsProviderType providerType,double distribution)
	{
		_getCount = getCount;
		_providerType = providerType;
		_distribution = distribution;
	}
	public IWeightsProvider Create(int layer)
	{
		switch (_providerType)
        {
			case WeightsProviderType.Random:
				return new RandomWeightsProvider(_getCount(layer), _distribution);
			case WeightsProviderType.GaussianRandom:
				return new GaussianRandomWeightsProvider(_getCount(layer), _distribution);
			case WeightsProviderType.LayerDependent:
				return new LayerDependentWeightsProvider(_getCount(layer), _distribution);
		}
		throw new ArgumentException();
	}
}