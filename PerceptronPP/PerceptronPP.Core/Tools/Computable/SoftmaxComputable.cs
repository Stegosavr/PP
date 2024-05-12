using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP.Core.Tools.Computable;

public class SoftmaxComputable //: IComputable //Only for last (output) layer
{
	public string Name { get; } = nameof(SoftmaxComputable);
	public static Matrix<double> Compute(Matrix<double> x)
	{
		var expArray = CreateMatrix.Dense<double>(x.RowCount, x.ColumnCount);
		for (var i = 0; i < expArray.ColumnCount; i++)
			expArray[0, i] = Math.Exp(x[0, i]);
		double sum = 0;
		for (var i = 0; i < expArray.ColumnCount; i++)
			sum += expArray[0, i];

		//var a = (expArray / sum);
		return expArray / sum;
	}


	public static Matrix<double> ComputeDerivative(Matrix<double> derMatrix)
	{
		var result = CreateMatrix.Dense<double>(derMatrix.RowCount, derMatrix.ColumnCount);
		for (int index = 0; index < derMatrix.ColumnCount; index++)
			for (int j = 0; j < derMatrix.ColumnCount; j++)
			{
				if (j != index)
					result[0, index] += -derMatrix[0,j] * derMatrix[0,index];
				else
					result[0, index] += derMatrix[0,index] * (1 - derMatrix[0,index]);
			}
		return result;
	}
}