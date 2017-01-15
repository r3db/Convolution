using System;

namespace Convolution
{
    internal struct ColorRaw
    {
        internal byte R;
        internal byte G;
        internal byte B;

        internal static ColorRaw FromRgb(byte r, byte g, byte b)
        {
            return new ColorRaw
            {
                R = r,
                G = g,
                B = b,
            };
        }
    }
}