using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

namespace SonicPrompter
{
    /// <summary>Holds one loop / oneshot statement plus parsed parameters.</summary>
    public sealed class Statement
    {
        public string kind;                  // "loop" | "oneshot"
        public string clip;
        public RangeOrValue starts_at  = RangeOrValue.Zero;
        public RangeOrValue duration   = RangeOrValue.Null; // loop-only
        public RangeOrValue every      = RangeOrValue.Zero; // oneshot
        public RangeOrValue volume     = new(1f);
        public RangeOrValue pitch      = new(1f);
        public float fade_in           = 0f;
        public float fade_out          = 0f;
    }

    /// <summary>Either a single value or a min..max range.</summary>
    public readonly struct RangeOrValue
    {
        public readonly float min, max;
        public readonly bool  isRange, isSet;
        public static readonly RangeOrValue Zero = new(0f);
        public static readonly RangeOrValue Null = default;

        public RangeOrValue(float v) { min = max = v; isRange = false; isSet = true; }
        public RangeOrValue(float a, float b){ min = a; max = b; isRange = true; isSet = true; }

        public float Sample() => !isSet ? 0f : (isRange ? Random.Range(min, max) : min);

        public static RangeOrValue Parse(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return Null;
            if (s.Contains(".."))
            {
                var p = s.Split("..");
                return new(float.Parse(p[0]), float.Parse(p[1]));
            }
            return new(float.Parse(s));
        }
    }

    /// <summary>Parses a .sp script into Statement objects.</summary>
    public static class SonicPrompterParser
    {
        // statement header + indented block
        private static readonly Regex StmtRx = new(
            @"^(?<kind>loop|oneshot)\s+""(?<clip>.+?)""\s*(?:every\s+(?<e1>\d+\.?\d*)\.\.(?<e2>\d+\.?\d*))?\s*:\s*\r?\n" +
            @"(?<block>(?:[ \t]+.*\r?\n?)*)",
            RegexOptions.Multiline | RegexOptions.IgnoreCase | RegexOptions.Compiled);

        // key = value inside block
        private static readonly Regex PropRx = new(
            @"^[ \t]+(?<key>\w+)\s*=\s*(?<val>[^\r\n#]+)",
            RegexOptions.Multiline | RegexOptions.Compiled);

        public static List<Statement> Parse(string scriptText)
        {
            var list = new List<Statement>();

            foreach (Match m in StmtRx.Matches(scriptText))
            {
                var s = new Statement
                {
                    kind = m.Groups["kind"].Value.ToLower(),
                    clip = m.Groups["clip"].Value.Trim()
                };

                if (m.Groups["e1"].Success)
                {
                    s.every = new RangeOrValue(
                        float.Parse(m.Groups["e1"].Value),
                        float.Parse(m.Groups["e2"].Value));
                }

                foreach (Match p in PropRx.Matches(m.Groups["block"].Value))
                {
                    string k = p.Groups["key"].Value.ToLower();
                    string v = p.Groups["val"].Value.Trim();

                    switch (k)
                    {
                        case "volume":    s.volume     = RangeOrValue.Parse(v); break;
                        case "pitch":     s.pitch      = RangeOrValue.Parse(v); break;
                        case "starts_at": s.starts_at  = RangeOrValue.Parse(v); break;
                        case "duration":  s.duration   = RangeOrValue.Parse(v); break;
                        case "fade_in":   s.fade_in    = float.Parse(v); break;
                        case "fade_out":  s.fade_out   = float.Parse(v); break;
                    }
                }

                list.Add(s);
            }

            return list;
        }

        /// <summary>Utility: clip path inside Resources/Audio/ (no extension).</summary>
        public static string PathFor(string clip)
            => $"Audio/{Path.GetFileNameWithoutExtension(clip)}";
    }
}
