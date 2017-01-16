using System;
using System.Drawing;
using System.Threading.Tasks;

namespace Convolution
{
    internal static class ConvolutionCpu
    {
        // Native GDI+ Bitmap!
        internal static Image Render1(Bitmap image, ConvolutionFilter filter)
        {
            var width = image.Width;
            var height = image.Height;

            var source = new FastBitmap(image);
            var result = new FastBitmap(image.Width, image.Height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var neighbours = GetNeighbours(x, y, width, height, (nx, ny, i) => (nx >= 0 && nx < width && ny >= 0 && ny < height) == false
                        ? Color.FromArgb(0, 0, 0)
                        : source.GetPixel(nx, ny));

                    result.SetPixel(x, y, filter.Compute(neighbours));
                }
            });

            return result.Bitmap;
        }

        // Custom Array!
        internal static Image Render2(Bitmap image, ConvolutionFilter filter)
        {
            var width = image.Width;
            var height = image.Height;

            var source = BitmapUtility.ToColorArray(image);
            var result = new ColorRaw[width * height];

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var offset = y * width + x;

                    var neighbours = GetNeighbours(x, y, width, height, (nx, ny, i) => (nx >= 0 && nx < width && ny >= 0 && ny < height) == false
                        ? ColorRaw.FromRgb(0, 0, 0)
                        : source[i]);

                    result[offset] = filter.Compute(neighbours);
                }
            });

            return BitmapUtility.FromColorArray(result, image.Width, image.Height);
        }

        // Helpers!
        private static T[] GetNeighbours<T>(int x, int y, int width, int height, Func<int, int, int, T> func)
        {
            const int offset = 1;

            var sy = y - offset;
            var sx = x - offset;
            var yl = y + offset;
            var xl = x + offset;

            var pixels = new T[9];
            var counter = 0;

            for (int ny = sy; ny <= yl; ++ny)
            {
                for (int nx = sx; nx <= xl; ++nx)
                {
                    pixels[counter++] = func(nx, ny, ny * width + nx);
                }
            }

            return pixels;
        }
    }
}