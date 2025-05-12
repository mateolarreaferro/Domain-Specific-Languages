using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

public class SonicPrompterRuntime : MonoBehaviour
{
    [Tooltip("Text asset containing .sp code")]
    [SerializeField] private TextAsset scriptFile;

    private readonly List<AudioSource> activeSources = new();

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

        ParseAndLaunch(scriptFile.text);
    }

    private void ParseAndLaunch(string code)
    {
        foreach (Match m in LoopStmt.Matches(code))
        {
            string clipName = m.Groups["clip"].Value;
            if (!float.TryParse(m.Groups["vol"].Value, out float volume))
                volume = 1f;

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

            activeSources.Add(src);
        }
    }

    public void StopAll()
    {
        foreach (var src in activeSources) src.Stop();
        activeSources.Clear();
    }

#if UNITY_EDITOR
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            StopAll();
            ParseAndLaunch(scriptFile.text);
            Debug.Log("[SonicPrompter] Script reloaded.");
        }
    }
#endif
}
