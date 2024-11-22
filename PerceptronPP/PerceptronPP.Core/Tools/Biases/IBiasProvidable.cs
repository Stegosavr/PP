namespace PerceptronPP.Core.Tools.Biases;

public interface IBiasProvidable
{
	public double GetBias(int layer, int neuron);
}