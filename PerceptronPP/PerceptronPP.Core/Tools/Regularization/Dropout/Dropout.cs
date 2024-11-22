using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP.Core.Tools;

public static class Dropout
{
    public static void DropActivations(Matrix<double> dropoutMatrix, DropoutData dropoutData)
	{
		if (dropoutData.Testing)
		{
			dropoutMatrix = dropoutMatrix * (1 - dropoutData.Probability);
			return;
		}

        var rnd = new Random();
		for (var i = 0; i < dropoutMatrix.ColumnCount; i++)
		{
			var r = rnd.NextDouble();
			if (r < dropoutData.Probability)
				dropoutMatrix[0, i] = 0;
		}
	}
}
