using System;
using System.Drawing;

namespace Convolution
{
    public sealed class PixelMatrix
    {
        private readonly int[] _filter;
        private readonly int _factor;
        private readonly int _offset;

        public PixelMatrix(int[] filter, int factor, int offset)
        {
            _filter = filter;
            _factor = factor;
            _offset = offset;
        }

        public static PixelMatrix EmptyFilter => new PixelMatrix(new[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, }, 1, 0);

        public static PixelMatrix GaussianBlurFilter => new PixelMatrix(new[] { 1, 2, 1, 2, 4, 2, 1, 2, 1, }, 16, 0);

        public static PixelMatrix SharpenFilter => new PixelMatrix(new[] { 0, -2, 0, -2, 11, -2, 0, -2, 0, }, 3, 0);

        public static PixelMatrix MeanRemovalFilter => new PixelMatrix(new[] { -1, -1, -1, -1, 9, -1, -1, -1, -1, }, 1, 0);

        public static PixelMatrix EmbossLaplascianFilter => new PixelMatrix(new[] { -1, 0, -1, 0, 4, 0, -1, 0, -1, }, 1, 127);

        public static PixelMatrix EdgeDetectFilter0 => new PixelMatrix(new[] { 1, 1, 1, 0, 0, 0, -1, -1, -1, }, 1, 127);

        public static PixelMatrix EdgeDetectFilter1 => new PixelMatrix(new[] { -1, 0, -1, 0, 4, 0, -1, 0, -1, }, 1, 0);

        public static PixelMatrix EdgeDetectFilter2 => new PixelMatrix(new[] { 1, 1, 1, 0, 0, 0, -1, -1, -1, }, 1, 0);

        public static PixelMatrix SobelX1 => new PixelMatrix(new[] { -1, -2, -1, 0, 0, 0, 1, 2, 1, }, 1, 0);

        public static PixelMatrix SobelX2 => new PixelMatrix(new[] { 1, 2, 1, 0, 0, 0, -1, -2, -1, }, 1, 0);

        public static PixelMatrix SobelY1 => new PixelMatrix(new[] { -1, 0, 1, -2, 0, 2, -1, 0, 1, }, 1, 0);

        public static PixelMatrix SobelY2 => new PixelMatrix(new[] { 1, 0, -1, 2, 0, -2, 1, 0, -1, }, 1, 0);

        public Color Compute(Color[] data)
        {
            float r = 0;
            float g = 0;
            float b = 0;

            for (int i = 0; i < _filter.Length; ++i)
            {
                r += this._filter[i] * data[i].R;
                g += this._filter[i] * data[i].G;
                b += this._filter[i] * data[i].B;
            }

            r = r / _factor + _offset;
            g = g / _factor + _offset;
            b = b / _factor + _offset;
            return Color.FromArgb(byte.MaxValue, Clamp(r), Clamp(g), Clamp(b));

        }

        private static byte Clamp(float value)
        {
            return (byte)(value < byte.MinValue ? byte.MinValue : value > byte.MaxValue ? byte.MaxValue : value);
        }
    }
}