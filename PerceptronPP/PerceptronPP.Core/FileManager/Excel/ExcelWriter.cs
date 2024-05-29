using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using OfficeOpenXml;
using OfficeOpenXml.Style;
using System.Text.RegularExpressions;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;
using PerceptronPP.Core.Tools.Computable;

namespace PerceptronPP.Core.FileManager.Excel;

public static class ExcelWriter
{
    public static void SaveNetworkParameters(Network network,IOptimizer optimizer,
        int batchSize, int batchSizeTo, double learningCoefficient, double learningTo,
        int trainingDataSize, double efficiency, double learningTimeInSeconds)
    {
        var type = network.GetType();
        var neuronsCount = String.Join('-',((int[])(type.GetField("_neuronCounts", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(network))).Select(e=>e.ToString()));
        var activationFunc = ((IComputable)(type.GetField("_activationComputable", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(network))).Name;

        ExcelWriter.SaveXLSX(@"..\..\..\..\LearningResults\parameters.xlsx", new ExcelLearningData[]
        {
            new ExcelLearningData
            {
                    HiddenLayers = neuronsCount,
                    ActivationFunction = activationFunc,
                    Optimizer = optimizer.Name,
                    BatchSize = batchSize.ToString(),
                    BatchSize2 = batchSizeTo == 0 ? null : batchSizeTo.ToString(),
                    LearningCoefficient = learningCoefficient.ToString(),
                    LearningCoefficient2 = learningTo == 0 ? null : learningTo.ToString(),
                    TrainingDataSize = trainingDataSize.ToString(),
                    PredictionEfficiency = efficiency.ToString(),
                    Time = learningTimeInSeconds.ToString()
            }
        });
    }

    public static Tuple<ExcelPackage,ExcelWorksheet> CreateXLSX(string path)
    {
        // If you use EPPlus in a noncommercial context
        // according to the Polyform Noncommercial license:

        ExcelPackage excel = new ExcelPackage();

        // name of the sheet 
        var workSheet = excel.Workbook.Worksheets.Add("Sheet1");

        var headers = typeof(ExcelLearningData)
            .GetFields()
            .Select(field => Regex
                .Replace(field.Name, @"([a-z])([A-Z])", "$1-$2")
                .ToLower());

        int i = 0;
        foreach (var name in headers)
        {
            workSheet.Cells[1, ++i].Value = name;
        }

        // By default, the column width is not  
        // set to auto fit for the content 
        // of the range, so we are using 
        // AutoFit() method here.  
        for (int j = 0; j < headers.Count(); j++)
            workSheet.Column(i+1).AutoFit();

        // file name with .xlsx extension  
        //string p_strPath = "H:\\geeksforgeeks.xlsx";

        // Create excel file on physical disk  
        FileStream objFileStrm = File.Create(path);
        objFileStrm.Close();

        return Tuple.Create(excel, workSheet);
    }


    public static void SaveXLSX(string path, ExcelLearningData[] data)
    {
        ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
        int recordIndex = 2;

        ExcelPackage excel;
        ExcelWorksheet workSheet;
        if (File.Exists(path))
        {
            excel = new ExcelPackage(new FileInfo(path));
            workSheet = excel.Workbook.Worksheets[0];
            var cell = workSheet.Cells[2, 1];
            while (workSheet.Cells[recordIndex, 1].Value != null)
                recordIndex++;
        }
        else
        {
            (excel, workSheet) = CreateXLSX(path);
        }


        // Inserting the article data into excel 
        // sheet by using the for each loop 
        // As we have values to the first row  
        // we will start with second row 

        foreach (var d in data)
        {
            d.GetType().GetFields()
            .Select(field => field.GetValue(d))
            .Select(value => value == null ? "" : value.ToString())
            .Select((value, i) => workSheet.Cells[recordIndex, i + 1].Value = value)
            .ToArray();
            recordIndex++;
        }

        // Write content to excel file  
        File.WriteAllBytes(path, excel.GetAsByteArray());
        //Close Excel package 
        excel.Dispose();
        //Console.ReadKey();
    }
}
