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

        ParseAndSync(scriptFile.text, fullReset: true);
    }

    /// <summary>
    /// Syncs AudioSources to the DSL.  If fullReset=true, stops everything first.
    /// </summary>
    private void ParseAndSync(string code, bool fullReset)
    {
        if (fullReset) StopAll();

        // 1. Parse DSL → collect desired set
        var desired = new Dictionary<string, float>(); // clip → volume
        foreach (Match m in LoopStmt.Matches(code))
        {
            string clip  = m.Groups["clip"].Value.Trim();
            float  vol   = float.TryParse(m.Groups["vol"].Value, out var v) ? v : 1f;
            desired[clip] = vol;                       // last one wins if duplicate
        }

        // 2. Update & spawn
        foreach (var kv in desired)
        {
            string clipName = kv.Key;
            float  vol      = kv.Value;

            if (activeSources.TryGetValue(clipName, out var src))
            {
                // already playing → just update parameters
                src.volume = vol;
            }
            else
            {
                // new loop → load & play
                string clipPath = $"Audio/{Path.GetFileNameWithoutExtension(clipName)}";
                AudioClip clip  = Resources.Load<AudioClip>(clipPath);
                if (clip == null)
                {
                    Debug.LogWarning($"Clip '{clipName}' not found in Resources/Audio");
                    continue;
                }

                var go  = new GameObject($"[SP] {clipName}");
                src     = go.AddComponent<AudioSource>();
                src.clip         = clip;
                src.loop         = true;
                src.volume       = vol;
                src.spatialBlend = 0f;
                src.Play();

                activeSources[clipName] = src;
                Debug.Log($"[SP] Started '{clipName}' vol={vol}");
            }
        }

        // 3. Stop & clean up anything no longer in the DSL
        var toRemove = new List<string>();
        foreach (var pair in activeSources)
            if (!desired.ContainsKey(pair.Key))
            {
                pair.Value.Stop();
                Destroy(pair.Value.gameObject);        // tidy up hierarchy
                toRemove.Add(pair.Key);
                Debug.Log($"[SP] Stopped '{pair.Key}' (removed from script)");
            }
        foreach (var key in toRemove) activeSources.Remove(key);
    }

    public void StopAll()
    {
        foreach (var src in activeSources.Values)
        {
            if (src) src.Stop();
            if (src) Destroy(src.gameObject);
        }
        activeSources.Clear();
    }

#if UNITY_EDITOR
    private void Update()
    {
        // R → incremental sync (add / update / remove)
        if (Input.GetKeyDown(KeyCode.R) && !Input.GetKey(KeyCode.LeftShift))
        {
            ParseAndSync(scriptFile.text, fullReset: false);
            Debug.Log("[SonicPrompter] Incremental sync.");
        }

        // Shift + R → full reset
        if (Input.GetKeyDown(KeyCode.R) && Input.GetKey(KeyCode.LeftShift))
        {
            ParseAndSync(scriptFile.text, fullReset: true);
            Debug.Log("[SonicPrompter] Full reset.");
        }
    }
#endif
}
