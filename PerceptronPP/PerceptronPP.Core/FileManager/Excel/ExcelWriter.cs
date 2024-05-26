using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using OfficeOpenXml;
using OfficeOpenXml.Style;
using System.Text.RegularExpressions;

namespace PerceptronPP.Core.FileManager.Excel;

public static class ExcelWriter
{
    //private ExcelLearningData _data { get; set; }

    //public ExcelWriter(ExcelLearningData data)
    //{
    //    _data = data;
    //}
    public static Tuple<ExcelPackage,ExcelWorksheet> CreateXLSX(string path)
    {
        // If you use EPPlus in a noncommercial context
        // according to the Polyform Noncommercial license:
        ExcelPackage.LicenseContext = LicenseContext.NonCommercial;


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


        // Header of the Excel sheet 
        //workSheet.Cells[1, 1].Value = "Optimizer";
        //workSheet.Cells[1, 2].Value = "Hidden layers";
        //workSheet.Cells[1, 3].Value = "Batch size";
        //workSheet.Cells[1, 5].Value = "Learning coefficient";
        //workSheet.Cells[1, 7].Value = "Training data size";
        //workSheet.Cells[1, 8].Value = "Prediction Efficiency";
        //workSheet.Cells[1, 9].Value = "Time";

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

        if (File.Exists(path))
            File.Delete(path);

        var (excel,workSheet) = CreateXLSX(path);

        // Inserting the article data into excel 
        // sheet by using the for each loop 
        // As we have values to the first row  
        // we will start with second row 
        int recordIndex = 2;

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
