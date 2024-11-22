using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core.Tools.Weights.Factory;

public class MapWeightsFactory : IWeightsFactory
{
	private readonly double[][,] _layersWeights;

	public MapWeightsFactory(params double[][,] layersWeights)
	{
		_layersWeights = layersWeights;
	}
	public IWeightsProvider Create(int layer)
	{
		return new ConstantWeightsProvider(_layersWeights[layer]);
	}
}