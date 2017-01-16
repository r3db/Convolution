using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;

namespace Convolution
{
    internal static class Program
    {
        private static void Main()
        {
            var image = new Bitmap(Image.FromFile(@"../../input.jpg"));

            Measure(() => EdgeDetectFilter0Cpu.Render1(image), "edge-detect-filter-0.cpu.1.png", false, "CPU: Using Native GDI+ Bitmap!");
            Measure(() => EdgeDetectFilter0Cpu.Render2(image), "edge-detect-filter-0.cpu.2.png", false, "CPU: Using Custom Array!");
            //Measure(() => EdgeDetectFilter0.RenderGpu3(image), "edge-detect-filter-0.gpu.3.png", true, "GPU: Parallel.For!");
            //Measure(() => EdgeDetectFilter0.RenderGpu1(image), "edge-detect-filter-0.gpu.1.png", true,  "GPU: Custom!");

            //Measure(() => Invert.RenderCpu1(image), "invert.cpu.1.png", false, "CPU: Using Native GDI+ Bitmap!");
            //Measure(() => Invert.Render2(image), "invert.cpu.2.png", false, "CPU: Using Custom Array!");
            //Measure(() => Invert.RenderGpu1(image), "invert.gpu.1.png", true, "GPU: Using Custom Array!");
            //Measure(() => Invert.RenderGpu3(image), "invert.gpu.3.png", true, "GPU: Parallel.For!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, bool isGpu, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = w => w.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format + "  (s)", w.Elapsed.TotalSeconds)
                : w.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", w.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", w.Elapsed.TotalMilliseconds * 1000);

            Action consoleColor = () =>
            {
                Console.ForegroundColor = isGpu
                    ? ConsoleColor.White
                    : ConsoleColor.Cyan;
            };

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            // Todo: Bandwith is not relevant for this problem!
            Func<Stopwatch, string> bandwidth = w => string.Format(CultureInfo.InvariantCulture, "{0,8:F4} GB/s", (result1.Width * result1.Height * 3) / (w.Elapsed.TotalMilliseconds * 1000000));

            Console.WriteLine(new string('-', 38));
            Console.WriteLine(description);
            consoleColor();
            Console.WriteLine("{0} - {1} [Cold]", formatElapsedTime(sw1), bandwidth(sw1));
            Console.ResetColor();
            result1.Save(fileName, ImageFormat.Png);

            var sw2 = Stopwatch.StartNew();
            func();
            sw2.Stop();
            consoleColor();
            Console.WriteLine("{0} - {1} [Warm]", formatElapsedTime(sw2), bandwidth(sw2));
            Console.ResetColor();
        }
    }
}