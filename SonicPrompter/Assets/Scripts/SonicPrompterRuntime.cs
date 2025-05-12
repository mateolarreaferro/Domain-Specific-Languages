using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SonicPrompter; 

public class SonicPrompterRuntime : MonoBehaviour
{
    [Tooltip("Text asset containing .sp code")]
    [SerializeField] private TextAsset scriptFile;

    private readonly List<AudioSource> spawned = new();
    private readonly List<Coroutine>   schedulers = new();

    private void Start()
    {
        if (!scriptFile) { Debug.LogError("No .sp file assigned."); return; }
        Sync(fullReset: true);
    }

#if UNITY_EDITOR
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R) && !Input.GetKey(KeyCode.LeftShift)) Sync(false);
        if (Input.GetKeyDown(KeyCode.R) &&  Input.GetKey(KeyCode.LeftShift)) Sync(true);
    }
#endif

    // ───────────────── sync & scheduling ─────────────────
    private void Sync(bool fullReset)
    {
        if (fullReset) HardReset();

        var stmts = SonicPrompterParser.Parse(scriptFile.text);
        foreach (var s in stmts) schedulers.Add(StartCoroutine(RunStmt(s)));

        Debug.Log($"[SP] Synced ({(fullReset ? "full" : "delta")}).");
    }

    private IEnumerator RunStmt(Statement s)
    {
        yield return new WaitForSeconds(s.starts_at.Sample());

        if (s.kind == "loop")
            yield return StartCoroutine(HandleLoop(s));
        else
            yield return StartCoroutine(HandleOneShot(s));
    }

    private IEnumerator HandleLoop(Statement s)
    {
        AudioSource src = SpawnSource(s);
        if (!src) yield break;

        if (s.duration.isSet)
            yield return StopAfter(src, s.duration.Sample(), s.fade_out);
    }

    private IEnumerator HandleOneShot(Statement s)
    {
        while (true)
        {
            AudioSource src = SpawnSource(s);
            if (!src) yield break;

            float wait = src.clip.length + s.fade_out + s.every.Sample();
            yield return new WaitForSeconds(wait);
        }
    }

    // ─────────── AudioSource helpers ───────────
    private AudioSource SpawnSource(Statement s)
    {
        var clip = Resources.Load<AudioClip>(SonicPrompterParser.PathFor(s.clip));
        if (!clip) { Debug.LogWarning($"Clip '{s.clip}' not found."); return null; }

        var go  = new GameObject($"[SP] {s.clip}");
        go.transform.SetParent(transform);
        var src = go.AddComponent<AudioSource>();
        spawned.Add(src);

        src.clip   = clip;
        src.loop   = (s.kind == "loop");
        src.volume = 0f;
        src.pitch  = s.pitch.Sample();
        src.Play();

        StartCoroutine(Fade(src, 0f, s.volume.Sample(), s.fade_in));
        return src;
    }

    private IEnumerator StopAfter(AudioSource src, float secs, float fadeOut)
    {
        yield return new WaitForSeconds(secs - fadeOut);
        yield return Fade(src, src.volume, 0f, fadeOut);
        if (src) src.Stop();
    }

    private static IEnumerator Fade(AudioSource src, float from, float to, float dur)
    {
        if (dur <= 0f) { if (src) src.volume = to; yield break; }
        float t = 0;
        while (t < dur && src)
        {
            src.volume = Mathf.Lerp(from, to, t / dur);
            t += Time.deltaTime;
            yield return null;
        }
        if (src) src.volume = to;
    }

    private void HardReset()
    {
        foreach (var co in schedulers) if (co != null) StopCoroutine(co);
        schedulers.Clear();

        foreach (var src in spawned)
            if (src) Destroy(src.gameObject);
        spawned.Clear();
    }
}
