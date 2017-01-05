using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Convolution
{
    internal static class Program
    {
        private static void Main()
        {
            var image = new Bitmap(Image.FromFile(@"../../input.jpg"));

            Measure(() => Invert.RenderCpu1(image), "invert.cpu.1.png", "CPU: Using Native GDI+ Bitmap!");
            Measure(() => Invert.RenderCpu2(image), "invert.cpu.2.png", "CPU: Using byte Array!");
            Measure(() => Invert.RenderGpu1(image), "invert.gpu.1.png", "GPU: Using byte Array!");
            Measure(() => Invert.RenderGpu3(image), "invert.gpu.3.png", "GPU: Parallel.For!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = (watch) => watch.Elapsed.TotalSeconds >= 1
                ? $"{watch.Elapsed.TotalSeconds}s"
                : $"{watch.ElapsedMilliseconds}ms";

            var sw1 = Stopwatch.StartNew();
            var bmp1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("{0} [Cold]", formatElapsedTime(sw1));
            Console.ResetColor();
            bmp1.Save(fileName, ImageFormat.Png);

            Console.WriteLine();

            var sw2 = Stopwatch.StartNew();
            func();
            sw2.Stop();
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} [Warm]", formatElapsedTime(sw2));
            Console.ResetColor();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine();
        }
    }
}