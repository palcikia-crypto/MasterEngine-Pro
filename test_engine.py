"""End-to-end engine test: generates a demo track, runs auto-mastering
(closed-loop) and per-stem mastering, prints before/after metrics."""
import numpy as np

from masterengine import analysis, dsp, master, stems

SR = 44100
DUR = 20.0
rng = np.random.default_rng(7)


def demo_track():
    n = int(SR * DUR)
    t = np.arange(n) / SR

    def adsr(env, a=0.01, d=0.1, s=0.6, r=0.2):
        e = np.ones_like(env)
        return e

    # kick
    kick = np.zeros(n)
    beat = int(0.5 * SR)
    for i in range(0, n - beat * 2, beat):
        f = np.geomspace(120, 42, int(0.12 * SR))
        ph = 2 * np.pi * np.cumsum(f) / SR
        idx = i + np.arange(len(f))
        kick[idx] = np.sin(ph) * np.exp(-idx * 0) * np.exp(-np.arange(len(f)) / (0.09 * SR))
    # hats / snare
    noise = rng.standard_normal(n) * 0.25
    snr = dsp.lr4_split(noise, SR, 1200, "hp")
    snare_env = np.zeros(n)
    for i in range(beat, n, beat * 2):
        e = np.exp(-np.arange(int(0.15 * SR)) / (0.05 * SR))
        snare_env[i:i + len(e)] = e
    drums = kick + 0.5 * snr * snare_env + 0.12 * dsp.lr4_split(noise, SR, 6000, "hp")
    # bass line
    notes = np.zeros(n)
    for i, bar in enumerate(range(0, n, beat * 4)):
        f0 = [55, 55, 73.4, 65.4][i % 4]
        ln = np.arange(min(beat * 4, n - bar)) / SR
        sig = np.sin(2 * np.pi * f0 * ln) + 0.4 * np.sin(2 * np.pi * 2 * f0 * ln)
        sig *= np.exp(-ln * 1.2)
        notes[bar:bar + len(sig)] += 0.5 * sig
    # chords pad (wide-ish: decorrelate channels)
    chords = np.zeros((n, 2))
    for i, bar in enumerate(range(0, n, beat * 8)):
        f0 = [220, 261.6, 293.7, 174.6][i % 4]
        ln = np.arange(min(beat * 8, n - bar)) / SR
        for mult in (1, 1.5, 2):
            lo = np.sin(2 * np.pi * f0 * mult * ln) * np.exp(-ln * 0.25) * (0.18 / mult)
            ro = np.sin(2 * np.pi * f0 * mult * ln * 1.003) * np.exp(-ln * 0.25) * (0.18 / mult)
            w = min(len(lo), n - bar)
            chords[bar:bar + w, 0] += lo[:w]
            chords[bar:bar + w, 1] += ro[:w]
    # lead melody
    lead = np.zeros((n, 2))
    for i, tt in enumerate(range(0, n, beat * 2)):
        f0 = [440, 494, 587, 523][i % 4]
        ln = np.arange(min(beat * 2, n - tt)) / SR
        sig = np.sin(2 * np.pi * f0 * ln) * np.exp(-ln * 2) * 0.25
        lead[tt:tt + len(sig), 0] += sig
        lead[tt:tt + len(sig), 1] += sig * 0.97

    if chords.ndim == 1:
        chords = np.stack([chords, chords], axis=1)
    if drums.ndim == 1:
        drums = np.stack([drums, drums], axis=1)
    if notes.ndim == 1:
        notes = np.stack([notes, notes], axis=1)
    mix = 0.9 * drums + 1.1 * notes + 0.8 * chords + lead
    # deliberately "unmastered": quiet-ish, dull, mono-ish, peaky
    mix *= 0.22
    mix[:, 0] = 0.5 * mix[:, 0] + 0.25 * mix[:, 1]
    return mix


def show(tag, f):
    print(f"  {tag:<10} {f['lufs']:7.2f} LUFS | {f['peak_dbtp']:6.2f} dBTP | {f['crest_db']:5.2f} dB crest | {f['stereo_width_db']:5.2f} dB width")


def main():
    x = demo_track()
    print("== demo track =", x.shape, SR, "Hz ==")
    fi = analysis.analyze(x, SR)
    show("before", fi)

    res, y = master.process(x, SR, refine_iters=7)
    fo = analysis.analyze(y, SR)
    show("after", fo)

    print("  LUFS reach:", round(fi["lufs"], 1), "→", round(fo["lufs"], 1),
          f"(target -14, err {abs(fo['lufs']+14):.2f})")
    print("  peak ok:", fo["peak_dbtp"] <= -0.5)
    assert fo["peak_dbtp"] <= -0.5, "true peak above ceiling"
    assert abs(fo["lufs"] + 14) < 1.0, "LUFS not converged"
    assert np.all(np.isfinite(y)) and np.abs(y).max() < 1.5

    print("== per-stem mastering ==")
    sm = stems.master_stems(x, SR, refine_iters=2)
    for name in sm["stems"]:
        show(name + " raw", sm["before"][name])
        show(name + " mst", sm["after"][name])
    print("== OK ==")


if __name__ == "__main__":
    main()
