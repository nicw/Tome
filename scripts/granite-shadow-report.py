#!/usr/bin/env python3
"""Render granite shadow comparison JSONs into one side-by-side HTML report.
Usage: python3 granite-shadow-report.py "~/Library/Application Support/Tome/GraniteShadow" [-o report.html]
Stdlib only (spec §7)."""
import argparse, difflib, html, json, pathlib, sys


def word_diff(a: str, b: str) -> tuple[float, str, str]:
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    left, right = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        at, bt = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
        if op == "equal":
            left.append(html.escape(at)); right.append(html.escape(bt))
        else:
            if at: left.append(f"<mark>{html.escape(at)}</mark>")
            if bt: right.append(f"<mark>{html.escape(bt)}</mark>")
    # Two empty strings (e.g. both sides blank) are trivially "equal" per
    # SequenceMatcher (ratio 1.0) — but an errored segment has empty
    # graniteText vs non-empty primaryText, which correctly scores as 100%
    # diff and sorts to the top.
    return 1 - sm.ratio(), " ".join(left), " ".join(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("report.html"))
    args = ap.parse_args()
    sessions = []
    unreadable = 0
    for p in sorted(args.dir.expanduser().glob("*.comparison.json")):
        try:
            sessions.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"warning: skipping malformed {p.name}: {e}", file=sys.stderr)
            unreadable += 1
    rows, agg = [], {"sessions": len(sessions), "segments": 0, "errored": 0,
                     "audio": 0.0, "wall": 0.0, "disagree": 0}
    for s in sessions:
        agg["segments"] += s["totals"]["segmentCount"]; agg["errored"] += s["totals"]["erroredCount"]
        agg["audio"] += s["totals"]["audioSeconds"]; agg["wall"] += s["totals"]["shadowWallClockSec"]
        for seg in s["segments"]:
            score, lh, rh = word_diff(seg["primaryText"], seg["graniteText"])
            if score > 0.05: agg["disagree"] += 1
            rows.append((score, s["session"]["sessionID"], s["session"]["primaryModel"],
                         s["totals"]["rtf"], s["incomplete"], seg, lh, rh))
    rows.sort(key=lambda r: -r[0])
    rtf = agg["wall"] / agg["audio"] if agg["audio"] else 0
    body = ["<h1>Granite shadow report</h1>",
            f"<p>{agg['sessions']} sessions &middot; {agg['segments']} segments &middot; "
            f"{agg['disagree']} disagreeing (&gt;5% word diff) &middot; {agg['errored']} errored &middot; "
            f"{unreadable} unreadable &middot; "
            f"aggregate shadow RTF {rtf:.3f}</p>",
            "<style>"
            "table{border-collapse:collapse;font-family:sans-serif}"
            "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
            "mark{background:#ffe08a}"
            ".incomplete{color:#b00020;font-weight:bold}"
            ".error-badge{background:#b00020;color:#fff;border-radius:3px;padding:1px 5px;"
            "font-size:0.8em;margin-right:4px}"
            "</style>",
            "<table>",
            "<tr><th>diff</th><th>session</th><th>t</th><th>primary</th><th>granite</th></tr>"]
    for score, sid, pmodel, srtf, incomplete, seg, lh, rh in rows:
        incomplete_marker = ' <span class="incomplete">INCOMPLETE</span>' if incomplete else ""
        error_badge = (f'<span class="error-badge">ERROR: {html.escape(seg["graniteError"])}</span><br>'
                       if seg.get("graniteError") else "")
        body.append(f"<tr><td>{score:.2f}</td><td>{html.escape(sid)}{incomplete_marker}<br><small>{html.escape(pmodel)}"
                    f" &middot; RTF {srtf:.3f}</small></td><td>{seg['startTime']:.0f}s</td>"
                    f"<td>{lh}</td><td>{error_badge}{rh}</td></tr>")
    body.append("</table>")
    args.out.write_text("\n".join(body))
    print(f"wrote {args.out} ({agg['segments']} segments)")


if __name__ == "__main__":
    main()
