using System;
using System.Drawing;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Convolution
{
    internal static class Invert
    {
        // CPU: Using Native GDI+ Bitmap!
        internal static Image RenderCpu1(Bitmap image)
        {
            var result = new FastBitmap(image);
            const int max = byte.MaxValue;

            Parallel.For(0, result.Height, y =>
            {
                for (var x = 0; x < result.Width; ++x)
                {
                    var color = result.GetPixel(x, y);
                    var newColor = Color.FromArgb(max - color.R, max - color.G, max - color.B);

                    result.SetPixel(x, y, newColor);
                }
            });

            return result.Bitmap;
        }

        // CPU: Using byte Array!
        internal static Image RenderCpu2(Bitmap image)
        {
            var width = image.Width;
            var height = image.Height;
            var result = FastBitmap.ToByteArray(image);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var offset = 3 * (y * width + x);
                    ComputeInvertAtOffset(result, offset);
                }
            });

            return FastBitmap.FromByteArray(result, image.Width, image.Height);
        }

        // GPU: Using byte Array!
        internal static Image RenderGpu1(Bitmap image)
        {
            var width = image.Width;
            var height = image.Height;
            var result = FastBitmap.ToByteArray(image);
            var lp = ComputeLaunchParameters(width, height);

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;
                var offset = 3 * (y * width + x);

                ComputeInvertAtOffset(result, offset);
            }, lp);

            return FastBitmap.FromByteArray(result, width, height);
        }

        // GPU: Parallel.For!
        internal static Image RenderGpu3(Bitmap image)
        {
            var result = FastBitmap.ToByteArray(image);

            Gpu.Default.For(0, image.Width * image.Height, i =>
            {
                var offset = 3 * i;
                ComputeInvertAtOffset(result, offset);
            });

            return FastBitmap.FromByteArray(result, image.Width, image.Height);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeInvertAtOffset(byte[] result, int offset)
        {
            if (offset < result.Length)
            {
                result[offset + 0] = (byte)(255 - result[offset + 0]);
                result[offset + 1] = (byte)(255 - result[offset + 1]);
                result[offset + 2] = (byte)(255 - result[offset + 2]);
            }
        }

        private static LaunchParam ComputeLaunchParameters(int width, int height)
        {
            const int threads = 32;
            return new LaunchParam(new dim3((width + (threads - 1)) / threads, (height + (threads - 1)) / threads), new dim3(threads, threads));
        }
    }
}