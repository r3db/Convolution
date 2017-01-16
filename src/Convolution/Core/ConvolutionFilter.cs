using System;
using System.Drawing;

namespace Convolution
{
    public sealed class ConvolutionFilter
    {
        private readonly float[] _filter;
        private readonly float _factor;
        private readonly float _offset;

        private ConvolutionFilter(float[] filter, int factor, int offset)
        {
            _filter = filter;
            _factor = factor;
            _offset = offset;
        }

        internal static ConvolutionFilter EmptyFilter => new ConvolutionFilter(new float[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, }, 1, 0);

        internal static ConvolutionFilter GaussianBlurFilter => new ConvolutionFilter(new float[] { 1, 2, 1, 2, 4, 2, 1, 2, 1, }, 16, 0);

        internal static ConvolutionFilter SharpenFilter => new ConvolutionFilter(new float[] { 0, -2, 0, -2, 11, -2, 0, -2, 0, }, 3, 0);

        internal static ConvolutionFilter MeanRemovalFilter => new ConvolutionFilter(new float[] { -1, -1, -1, -1, 9, -1, -1, -1, -1, }, 1, 0);

        internal static ConvolutionFilter EmbossLaplascianFilter => new ConvolutionFilter(new float[] { -1, 0, -1, 0, 4, 0, -1, 0, -1, }, 1, 127);

        internal static ConvolutionFilter EdgeDetectFilter0 => new ConvolutionFilter(new float[] { 1, 1, 1, 0, 0, 0, -1, -1, -1, }, 1, 127);

        internal static ConvolutionFilter EdgeDetectFilter1 => new ConvolutionFilter(new float[] { -1, 0, -1, 0, 4, 0, -1, 0, -1, }, 1, 0);

        internal static ConvolutionFilter EdgeDetectFilter2 => new ConvolutionFilter(new float[] { 1, 1, 1, 0, 0, 0, -1, -1, -1, }, 1, 0);

        internal static ConvolutionFilter SobelX1 => new ConvolutionFilter(new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1, }, 1, 0);

        internal static ConvolutionFilter SobelX2 => new ConvolutionFilter(new float[] { 1, 2, 1, 0, 0, 0, -1, -2, -1, }, 1, 0);

        internal static ConvolutionFilter SobelY1 => new ConvolutionFilter(new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1, }, 1, 0);

        internal static ConvolutionFilter SobelY2 => new ConvolutionFilter(new float[] { 1, 0, -1, 2, 0, -2, 1, 0, -1, }, 1, 0);

        // Todo: Remove!
        internal float[] Filter => _filter;
        internal float Factor => _factor;
        internal float Offset => _offset;

        internal Color Compute(Color[] data)
        {
            float r = 0;
            float g = 0;
            float b = 0;

            for (int i = 0; i < _filter.Length; ++i)
            {
                r += _filter[i] * data[i].R;
                g += _filter[i] * data[i].G;
                b += _filter[i] * data[i].B;
            }

            r = r / _factor + _offset;
            g = g / _factor + _offset;
            b = b / _factor + _offset;

            return Color.FromArgb(Clamp(r), Clamp(g), Clamp(b));
        }

        internal ColorRaw Compute(ColorRaw[] data)
        {
            float r = 0;
            float g = 0;
            float b = 0;

            for (int i = 0; i < _filter.Length; ++i)
            {
                r += _filter[i] * data[i].R;
                g += _filter[i] * data[i].G;
                b += _filter[i] * data[i].B;
            }

            r = r / _factor + _offset;
            g = g / _factor + _offset;
            b = b / _factor + _offset;

            return ColorRaw.FromRgb(Clamp(r), Clamp(g), Clamp(b));
        }

        private static byte Clamp(float value)
        {
            return (byte)(value < byte.MinValue ? byte.MinValue : value > byte.MaxValue ? byte.MaxValue : value);
        }
    }
}