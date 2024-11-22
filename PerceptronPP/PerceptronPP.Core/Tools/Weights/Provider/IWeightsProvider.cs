namespace PerceptronPP.Core.Tools.Weights.Provider;

public interface IWeightsProvider
{
	public double GetWeight(int current, int next);
}