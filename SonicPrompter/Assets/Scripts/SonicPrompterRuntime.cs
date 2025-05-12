using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

public class SonicPrompterRuntime : MonoBehaviour
{
    [Tooltip("Text asset containing .sp code")]
    [SerializeField] private TextAsset scriptFile;

    // clip name  →  AudioSource
    private readonly Dictionary<string, AudioSource> activeSources = new();

    private static readonly Regex LoopStmt =
        new(
            @"loop\s+""(?<clip>.+?)""\s*:\s*(?:\r?\n\s*volume\s*=\s*(?<vol>\d*\.?\d+))?",
            RegexOptions.Compiled | RegexOptions.IgnoreCase
        );

    private void Start()
    {
        if (scriptFile == null)
        {
            Debug.LogError("No SonicPrompter script assigned!");
            return;
        }

        ParseAndLaunch(scriptFile.text, fullReset: true);
    }

    /// <summary>
    /// Parses DSL and spawns/updates AudioSources.
    /// If fullReset is true, stops everything first.
    /// </summary>
    private void ParseAndLaunch(string code, bool fullReset)
    {
        if (fullReset) StopAll();

        foreach (Match m in LoopStmt.Matches(code))
        {
            string clipName = m.Groups["clip"].Value.Trim();
            float  volume   = float.TryParse(m.Groups["vol"].Value, out var v) ? v : 1f;

            // Already playing?
            if (activeSources.TryGetValue(clipName, out var existing))
            {
                existing.volume = volume;               // just update param
                continue;                               // don’t restart clip
            }

            // New loop → load clip + create source
            string clipPath = $"Audio/{Path.GetFileNameWithoutExtension(clipName)}";
            AudioClip clip  = Resources.Load<AudioClip>(clipPath);
            if (clip == null)
            {
                Debug.LogWarning($"Clip '{clipName}' not found in Resources/Audio");
                continue;
            }

            var go  = new GameObject($"[SP] {clipName}");
            var src = go.AddComponent<AudioSource>();

            src.clip         = clip;
            src.loop         = true;
            src.volume       = volume;
            src.spatialBlend = 0f; // 2D
            src.Play();

            activeSources[clipName] = src;
            Debug.Log($"[SP] Started '{clipName}' vol={volume}");
        }
    }

    public void StopAll()
    {
        foreach (var src in activeSources.Values) src.Stop();
        activeSources.Clear();
    }

#if UNITY_EDITOR
    private void Update()
    {
        // R  → incremental reload (add / tweak, but don't stop others)
        if (Input.GetKeyDown(KeyCode.R) && !Input.GetKey(KeyCode.LeftShift))
        {
            ParseAndLaunch(scriptFile.text, fullReset: false);
            Debug.Log("[SonicPrompter] Incremental reload.");
        }

        // Shift + R  → full reset & reload (old behaviour)
        if (Input.GetKeyDown(KeyCode.R) && Input.GetKey(KeyCode.LeftShift))
        {
            ParseAndLaunch(scriptFile.text, fullReset: true);
            Debug.Log("[SonicPrompter] Full reset + reload.");
        }
    }
#endif
}
