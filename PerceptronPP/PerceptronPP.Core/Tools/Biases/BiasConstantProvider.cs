namespace PerceptronPP.Core.Tools.Biases;

public class BiasConstantProvider : IBiasProvidable
{
	private readonly int _constant;

	public BiasConstantProvider(int constant)
	{
		_constant = constant;
	}
	
	public double GetBias(int _, int __)
	{
		return _constant;
	}
}