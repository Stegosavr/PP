namespace PerceptronPP;

internal static class Tools
{
	public static void PrintDoubleArray(double[] array)
	{
		if (array.Length == 0) return;
		Console.Write(array[0]);
		foreach (var number in array.Skip(1))
			Console.Write($", {number}");
		Console.WriteLine();
	}
}