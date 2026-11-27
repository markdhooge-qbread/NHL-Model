"""Free DraftKings odds feeds (no API key, no credits).

  * Game lines (moneyline, total, puck line): ESPN's public NHL scoreboard, whose odds come from DraftKings.
  * Player props (SOG, points, anytime goal): DraftKings' own public sportsbook JSON (the feed its website uses).

Neither is an official API, so both can change or be blocked without warning. Every function returns
rows in the lines-file format (game, market, player, line, over_odds, under_odds, source) plus an error
string, and never raises, so the app can fall back to the screenshot lines file.
"""
import datetime
import re
import unicodedata
from zoneinfo import ZoneInfo

import requests

EST = ZoneInfo("America/New_York")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
DK_LEAGUE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42133"  # NHL
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
                   "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://sportsbook.draftkings.com",
    "Referer": "https://sportsbook.draftkings.com/",
}
SRC_ESPN, SRC_DK = "ESPN", "DK"
ROW_COLS = ["game", "market", "player", "line", "over_odds", "under_odds", "source"]

# NHL abbreviations the app uses, keyed by the short codes other sites use.
ABBR_ALIASES = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "UTAH": "UTA", "UTH": "UTA",
                "WAS": "WSH", "MON": "MTL", "NAS": "NSH", "CLB": "CBJ", "LV": "VGK", "VEG": "VGK"}
NHL_ABBRS = {"ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN",
             "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA",
             "VAN", "VGK", "WSH", "WPG"}
NICKNAMES = {
    "ducks": "ANA", "bruins": "BOS", "sabres": "BUF", "flames": "CGY", "hurricanes": "CAR", "blackhawks": "CHI",
    "avalanche": "COL", "bluejackets": "CBJ", "stars": "DAL", "redwings": "DET", "oilers": "EDM",
    "panthers": "FLA", "kings": "LAK", "wild": "MIN", "canadiens": "MTL", "predators": "NSH", "devils": "NJD",
    "islanders": "NYI", "rangers": "NYR", "senators": "OTT", "flyers": "PHI", "penguins": "PIT", "sharks": "SJS",
    "kraken": "SEA", "blues": "STL", "lightning": "TBL", "mapleleafs": "TOR", "mammoth": "UTA",
    "hockeyclub": "UTA", "canucks": "VAN", "goldenknights": "VGK", "capitals": "WSH", "jets": "WPG",
}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def team_abbr(name=None, abbr=None):
    """NHL abbreviation from a full name ('Toronto Maple Leafs'), a DK name ('TOR Maple Leafs') or a short code."""
    if abbr:
        a = str(abbr).upper().strip()
        a = ABBR_ALIASES.get(a, a)
        if a in NHL_ABBRS:
            return a
    if name:
        n = _norm(name)
        for nick, ab in sorted(NICKNAMES.items(), key=lambda kv: -len(kv[0])):
            if n.endswith(nick):
                return ab
        first = str(name).split()[0].upper() if str(name).split() else ""
        first = ABBR_ALIASES.get(first, first)
        if first in NHL_ABBRS:
            return first
    return None


def american(v):
    """'-130', '+115', '−130' (unicode minus), 'EVEN', 115 -> int American odds, else None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v) if abs(v) >= 100 else None
    s = str(v).strip().replace("−", "-").replace("–", "-").replace(",", "")
    if s.upper() in ("EVEN", "EV", "PK"):
        return 100
    try:
        o = int(float(s.replace("+", "")))
    except ValueError:
        return None
    return o if abs(o) >= 100 else None


def decimal_to_american(d):
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _implied(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def _num(v):
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace("−", "-"))
    return float(m.group()) if m else None


def _et_date(iso):
    try:
        return datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(EST).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _row(game, market, player="", line=None, over=None, under=None, source=""):
    return {"game": game, "market": market, "player": player or "",
            "line": "" if line is None else f"{line:g}",
            "over_odds": "" if over is None else f"{over:+d}",
            "under_odds": "" if under is None else f"{under:+d}", "source": source}


# ----------------------------------------------------------------------------
# ESPN: game lines
# ----------------------------------------------------------------------------
def _pick(d, *paths):
    """First non-empty value along any of the dotted paths."""
    for p in paths:
        cur = d
        for k in p.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur not in (None, ""):
            return cur
    return None


def parse_espn(js, date_str):
    rows = []
    for ev in js.get("events", []) or []:
        comps = ev.get("competitions") or [{}]
        comp = comps[0]
        if _et_date(comp.get("date") or ev.get("date")) not in (date_str, None):
            continue
        teams = {}
        for c in comp.get("competitors", []) or []:
            t = c.get("team", {}) or {}
            teams[c.get("homeAway")] = team_abbr(t.get("displayName") or t.get("name"), t.get("abbreviation"))
        away, home = teams.get("away"), teams.get("home")
        if not (away and home):
            continue
        odds_list = comp.get("odds") or []
        if not odds_list:
            continue
        o = next((x for x in odds_list if "draftkings" in _norm(_pick(x, "provider.name") or "")), odds_list[0])
        game = f"{away}@{home}"
        # Moneyline
        for side, team in (("away", away), ("home", home)):
            ml = american(_pick(o, f"moneyline.{side}.close.odds", f"moneyline.{side}.current.odds",
                                f"{side}TeamOdds.moneyLine", f"{side}TeamOdds.current.moneyLine.american",
                                f"moneyline.{side}.open.odds"))
            if ml is not None:
                rows.append(_row(game, "ML", team, None, ml, None, SRC_ESPN))
        # Total
        line = _num(_pick(o, "total.over.close.line", "total.over.current.line", "overUnder", "total.over.open.line"))
        over = american(_pick(o, "total.over.close.odds", "total.over.current.odds", "overOdds", "current.over.american"))
        under = american(_pick(o, "total.under.close.odds", "total.under.current.odds", "underOdds",
                               "current.under.american"))
        if line is not None and (over is not None or under is not None):
            rows.append(_row(game, "TOTAL", "", line, over, under, SRC_ESPN))
        # Puck line (saved for reference; the model doesn't price full-game puck lines yet)
        for side, team in (("away", away), ("home", home)):
            pl = _num(_pick(o, f"pointSpread.{side}.close.line", f"pointSpread.{side}.current.line"))
            po = american(_pick(o, f"pointSpread.{side}.close.odds", f"pointSpread.{side}.current.odds",
                                f"{side}TeamOdds.spreadOdds"))
            if pl is None and _pick(o, "spread") is not None:
                sp = _num(o.get("spread"))  # ESPN's spread is from the home side
                pl = sp if side == "home" else (-sp if sp is not None else None)
            if pl is not None and po is not None:
                rows.append(_row(game, "PL", team, pl, po, None, SRC_ESPN))
    return rows


def fetch_espn(date_str, timeout=15):
    """(rows, error). Error is None on success, even if ESPN hasn't posted odds yet (then rows is empty)."""
    try:
        r = requests.get(ESPN_SCOREBOARD, params={"dates": date_str.replace("-", ""), "limit": 50},
                         headers={"User-Agent": BROWSER_HEADERS["User-Agent"]}, timeout=timeout)
        if r.status_code != 200:
            return [], f"ESPN HTTP {r.status_code}"
        return parse_espn(r.json(), date_str), None
    except Exception as exc:  # noqa: BLE001
        return [], f"ESPN: {type(exc).__name__}"


# ----------------------------------------------------------------------------
# DraftKings: player props
# ----------------------------------------------------------------------------
PROP_KINDS = {
    # market: (subcategory name must match, ...and must not match)
    "Goal": (r"anytime\s*(goal\s*)?scorer|anytime goal", r"period|1st|2nd|3rd|first|last|team|power"),
    "SOG": (r"shots on goal|^shots$|player shots", r"period|1st|2nd|3rd|team|goalie|save|blocked|attempt"),
    "Pts": (r"^points|player points|points o/?u|points milestones", r"period|1st|2nd|3rd|team|power|goalie"),
}


def _walk(obj, key):
    """Every list stored under `key` anywhere in a JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, list):
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _dk_get(url, timeout):
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def dk_prop_subcategories(league_js, category_jsons):
    """[(market, category_id, subcategory_id, name)] for the prop subcategories we price."""
    found, seen = [], set()
    for js in [league_js, *category_jsons]:
        for lst in _walk(js, "subcategories"):
            for sc in lst:
                if not isinstance(sc, dict):
                    continue
                name = str(sc.get("name", ""))
                sid, cid = sc.get("id"), sc.get("categoryId")
                for mkt, (want, avoid) in PROP_KINDS.items():
                    if re.search(want, name, re.I) and not re.search(avoid, name, re.I) and (sid, mkt) not in seen:
                        seen.add((sid, mkt))
                        found.append((mkt, cid, sid, name))
    return found


def dk_prop_categories(league_js):
    """Category ids whose names look like player prop pages (goalscorer, shots, points)."""
    out = []
    for lst in _walk(league_js, "categories"):
        for c in lst:
            if isinstance(c, dict) and re.search(r"goal\s*scorer|shot|point|player", str(c.get("name", "")), re.I) \
                    and not re.search(r"goalie|team|period", str(c.get("name", "")), re.I):
                out.append(c.get("id"))
    return [c for c in dict.fromkeys(out) if c is not None]


def _sel_odds(s):
    o = american(_pick(s, "displayOdds.american", "odds.american", "americanOdds", "oddsAmerican"))
    return o if o is not None else decimal_to_american(_pick(s, "trueOdds", "displayOdds.decimal", "oddsDecimal"))


def _sel_player(s, market):
    for p in s.get("participants") or []:
        if isinstance(p, dict) and str(p.get("type", "")).lower() == "player" and p.get("name"):
            return p["name"]
    label = str(s.get("label", ""))
    if label and not re.match(r"^(over|under|yes|no|\d+\+?)$", label, re.I):
        return re.sub(r"\s+(over|under)$", "", label, flags=re.I)
    name = str(market.get("name", ""))
    return re.sub(r"\s*(-|–)?\s*(shots on goal|points|anytime goalscorer|goals?)(\s*o/?u)?\s*$", "", name, flags=re.I) or None


def parse_dk(js, market_kind, date_str):
    """Rows for one DK subcategory response (events / markets / selections)."""
    events = {}
    for lst in _walk(js, "events"):
        for e in lst:
            if not isinstance(e, dict) or e.get("id") is None:
                continue
            if _et_date(e.get("startEventDate")) not in (date_str, None):
                continue
            side = {}
            for p in e.get("participants") or []:
                role = str(p.get("venueRole", "")).lower()
                if role in ("home", "away"):
                    side[role] = team_abbr(p.get("name"), p.get("shortName") or p.get("abbreviation"))
            if not side.get("home") or not side.get("away"):
                m = re.match(r"(.+?)\s+(?:@|at|vs\.?)\s+(.+)", str(e.get("name", "")))
                if m:
                    side = {"away": team_abbr(m.group(1)), "home": team_abbr(m.group(2))}
            if side.get("home") and side.get("away"):
                events[e["id"]] = f"{side['away']}@{side['home']}"
    markets = {}
    for lst in _walk(js, "markets"):
        for m in lst:
            if isinstance(m, dict) and m.get("id") is not None and m.get("eventId") in events:
                if re.search(r"period|1st|2nd|3rd", str(m.get("name", "")), re.I):
                    continue
                markets[m["id"]] = m
    ou, ms = {}, {}  # O/U lines and N+ milestones, keyed (game, player)
    for lst in _walk(js, "selections"):
        for s in lst:
            if not isinstance(s, dict):
                continue
            m = markets.get(s.get("marketId"))
            odds = _sel_odds(s)
            if not m or odds is None:
                continue
            game, player = events[m["eventId"]], _sel_player(s, m)
            if not player:
                continue
            if market_kind == "Goal":
                if re.match(r"^no$", str(s.get("label", "")), re.I):
                    continue
                ou.setdefault((game, player), {"line": 0.5, "over": None, "under": None})["over"] = odds
                continue
            outcome = str(s.get("outcomeType") or s.get("label") or "").lower()
            pts = _num(s.get("points"))
            mil = re.match(r"^(\d+)\+$", str(s.get("label", "")).strip())
            if outcome in ("over", "under") and pts is not None:
                rec = ou.setdefault((game, player, pts), {"line": pts, "over": None, "under": None})
                rec[outcome] = odds
            elif mil:
                ms.setdefault((game, player), []).append({"line": int(mil.group(1)) - 0.5, "over": odds, "under": None})
    rows, have = [], set()
    # One main O/U line per player: the most balanced one if DK lists alternates.
    best = {}
    for key, rec in ou.items():
        gp = key[:2]
        bal = abs(abs(rec["over"] or 999) - abs(rec["under"] or 999)) if market_kind != "Goal" else 0
        if gp not in best or bal < best[gp][0]:
            best[gp] = (bal, rec)
    for (game, player), (_, rec) in best.items():
        rows.append(_row(game, market_kind, player, rec["line"], rec["over"], rec["under"], SRC_DK))
        have.add((game, player))
    for (game, player), recs in ms.items():
        if (game, player) in have:
            continue
        rec = min(recs, key=lambda r: abs(_implied(r["over"]) - 0.5))
        rows.append(_row(game, market_kind, player, rec["line"], rec["over"], None, SRC_DK))
    return rows


def fetch_dk_props(date_str, timeout=15):
    """(rows, error). Discovers the prop subcategory ids from DK's league page, then reads each one."""
    try:
        league = _dk_get(DK_LEAGUE, timeout)
    except Exception as exc:  # noqa: BLE001
        return [], f"DK blocked or down ({exc})"
    cats = []
    for cid in dk_prop_categories(league)[:6]:
        try:
            cats.append(_dk_get(f"{DK_LEAGUE}/categories/{cid}", timeout))
        except Exception:  # noqa: BLE001
            continue
    subs = dk_prop_subcategories(league, cats)
    if not subs:
        return [], "DK props not found (feed format may have changed)"
    rows, errs, seen = [], [], set()
    for mkt, cid, sid, _name in subs:
        if cid is None:
            continue
        try:
            js = _dk_get(f"{DK_LEAGUE}/categories/{cid}/subcategories/{sid}", timeout)
        except Exception as exc:  # noqa: BLE001
            errs.append(str(exc))
            continue
        for r in parse_dk(js, mkt, date_str):
            k = (r["game"], r["market"], _norm(r["player"]))
            if k not in seen:
                seen.add(k)
                rows.append(r)
    if not rows and errs:
        return [], f"DK props failed ({errs[0]})"
    return rows, None
