using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core.Tools.Weights.Factory;

public class ConstantWeightFactory : IWeightsFactory
{
	private readonly double[][,] _layersWeights;

	public ConstantWeightFactory(params double[][,] layersWeights)
	{
		_layersWeights = layersWeights;
	}
	public IWeightsProvider Create(int layer)
	{
		return new ConstantWeightsProvider(_layersWeights[layer]);
	}
}