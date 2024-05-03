namespace PerceptronPP.Core.Errors;

public class IncorrectNeuronCountException : Exception
{
	public IncorrectNeuronCountException() : base("Number of neurons on this layer is different") { }
}