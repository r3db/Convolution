using System;
using System.Drawing;
using Alea;
using Alea.Parallel;

namespace Convolution
{
    // Todo: Check this out: https://www.evl.uic.edu/sjames/cs525/final.html
    internal static class EdgeDetectFilter0Gpu
    {
        // Alea Parallel.For!
        internal static Image Render1(Bitmap image)
        {
            var gpu = Gpu.Default;

            var width = image.Width;
            var array = BitmapUtility.ToColorArray(image);

            var matrix = PixelMatrix.EdgeDetectFilter0;
            var mFilter = matrix.Filter;
            var mFactor = matrix.Factor;
            var mOffset = matrix.Offset;

            var inputMemory = gpu.ArrayGetMemory(array, true, false);
            var inputDevPtr = new deviceptr<ColorRaw>(inputMemory.Handle);

            var resultLength = array.Length;
            var resultMemory = Gpu.Default.AllocateDevice<ColorRaw>(resultLength);
            var resultDevPtr = new deviceptr<ColorRaw>(resultMemory.Handle);

            gpu.For(0, resultLength, i =>
            {
                if (i < resultLength)
                {
                    ComputeEdgeDetectFilter0AtOffset(inputDevPtr, resultDevPtr, resultLength, mFilter, mFactor, mOffset, i, width);
                }
            });

            return BitmapUtility.FromColorArray(Gpu.CopyToHost(resultMemory), image.Width, image.Height);
        }

        // Helpers!
        private static void ComputeEdgeDetectFilter0AtOffset(deviceptr<ColorRaw> input, deviceptr<ColorRaw> result, int resultLength, float[] filter, float mFactor, float mOffset, int i, int width)
        {
            var cx = i % width;
            var cy = i / width;

            var sx = cx - 1;
            var sy = cy - 1;
            var ex = cx + 1;
            var ey = cy + 1;

            var r = 0f;
            var g = 0f;
            var b = 0f;

            var filterIndex = 0;

            for (var y = sy; y <= ey; ++y)
            {
                for (var x = sx; x <= ex; ++x)
                {
                    var offset = y * width + x;

                    var pixel = offset < resultLength
                        ? input[offset]
                        : ColorRaw.FromRgb(0, 0, 0);

                    var currentFilter = filter[filterIndex++];

                    r += pixel.R * currentFilter;
                    g += pixel.G * currentFilter;
                    b += pixel.B * currentFilter;
                }
            }

            r = r / mFactor + mOffset;
            g = g / mFactor + mOffset;
            b = b / mFactor + mOffset;

            result[i] = ColorRaw.FromRgb(Clamp(r), Clamp(g), Clamp(b));
        }

        private static byte Clamp(float value)
        {
            var result = value < byte.MinValue
                ? byte.MinValue
                : value > byte.MaxValue
                    ? byte.MaxValue
                    : value;

            return (byte)result;
        }
    }
}