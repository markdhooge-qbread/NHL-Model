"""Runs the whole Streamlit app against fake MoneyPuck / NHL / GitHub responses (no network).

Checks the app renders without errors, saves tonight's Best bets candidates with the top pick before
the first puck drop, and appends yesterday's top pick to the CLV log using that day's saved feed lines.
"""
import base64
import datetime
import hashlib
import json
import os
import sys
from zoneinfo import ZoneInfo

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import fixtures as F  # noqa: E402

EST = ZoneInfo("America/New_York")


class Resp:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})
        self.content = self.text.encode()

    def json(self):
        return self._payload if self._payload is not None else json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def _sched_game(gid, away, home, start, state):
    return {"id": gid, "gameType": 2, "gameState": state, "startTimeUTC": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "awayTeam": {"abbrev": away, "score": 4 if state == "OFF" else None},
            "homeTeam": {"abbrev": home, "score": 3 if state == "OFF" else None},
            "gameOutcome": {"lastPeriodType": "REG"}}


@pytest.fixture
def fake_world(monkeypatch):
    now = datetime.datetime.now(EST)
    today, yday = now.date().isoformat(), (now.date() - datetime.timedelta(days=1)).isoformat()
    utc = datetime.timezone.utc
    later = (now + datetime.timedelta(hours=3)).astimezone(utc)
    games = {
        today: [_sched_game(2026020500, "DAL", "STL", later, "FUT"), _sched_game(2026020501, "TOR", "BOS", later, "FUT")],
        yday: [_sched_game(2026020400, "EDM", "TBL", (now - datetime.timedelta(days=1)).astimezone(utc), "OFF")],
    }
    csvs = {"teams": F.teams_df().to_csv(index=False), "goalies": F.goalies_df().to_csv(index=False),
            "skaters": F.skaters_df().to_csv(index=False)}
    store = {
        "picks/top_picks.csv": ("date,pick,market,game,odds,p,imp,edge,saved_at\n"
                                f"{yday},EDM @ TBL O 5.5,Total,EDM @ TBL,-105,0.58,0.5,8.0,12:00 ET\n"),
        f"lines/auto/{yday}.csv": ("game,market,player,line,over_odds,under_odds,source\n"
                                   "EDM@TBL,TOTAL,,5.5,-120,+100,ESPN\nEDM@TBL,ML,EDM,,-130,,ESPN\nEDM@TBL,ML,TBL,,+110,,ESPN\n"),
        f"lines/auto/{today}.csv": ("game,market,player,line,over_odds,under_odds,source\n"
                                    "DAL@STL,ML,DAL,,+250,,ESPN\nDAL@STL,ML,STL,,-300,,ESPN\n"
                                    "DAL@STL,TOTAL,,5.5,+150,-180,ESPN\nTOR@BOS,ML,TOR,,+220,,ESPN\n"
                                    "TOR@BOS,ML,BOS,,-260,,ESPN\nTOR@BOS,TOTAL,,6.5,-200,+165,ESPN\n"),
    }

    def fake_get(url, *a, **kw):
        if "moneypuck.com" in url:
            return Resp(text=csvs[url.rsplit("/", 1)[1][:-4]])
        if "/schedule/" in url:
            d = url.rsplit("/", 1)[1]
            return Resp(payload={"gameWeek": [{"date": d, "games": games.get(d, [])}]})
        if "/score/" in url:
            return Resp(payload={"games": []})
        if url.endswith("/boxscore"):
            gid = int(url.split("/")[-2])
            g = next((x for v in games.values() for x in v if x["id"] == gid), None)
            if g and g["gameState"] == "OFF":
                return Resp(payload={"gameState": "OFF", "awayTeam": {"score": 4}, "homeTeam": {"score": 3},
                                     "playerByGameStats": {}})
            return Resp(payload={"gameState": "FUT"})
        if url.endswith("/landing"):
            return Resp(payload={})
        raise requests.ConnectionError("offline in tests")

    def fake_request(method, url, *a, **kw):
        path = url.split("/repos/", 1)[1].split("/", 2)[2]
        if path.startswith("branches/"):
            return Resp(payload={"name": "picks-data"})
        if not path.startswith("contents/"):
            return Resp(status=404)
        p = path[len("contents/"):]
        if method == "PUT":
            store[p] = base64.b64decode(kw["json"]["content"]).decode()
            return Resp(status=201, payload={})
        if p in store:
            return Resp(payload={"content": base64.b64encode(store[p].encode()).decode(),
                                 "sha": hashlib.sha1(store[p].encode()).hexdigest()})
        kids = [k for k in store if k.startswith(p + "/")]
        if kids:
            return Resp(payload=[{"name": k.rsplit("/", 1)[1]} for k in kids])
        return Resp(status=404)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "request", fake_request)
    return store, today, yday


def test_app_runs_and_logs_clv(fake_world):
    store, today, yday = fake_world
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180)
    at.secrets["GITHUB_TOKEN"] = "test"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Closing line value" in text
    # tonight: candidates saved with the top pick
    assert f"picks/candidates/{today}.csv" in store
    cands = store[f"picks/candidates/{today}.csv"]
    assert "is_top" in cands.splitlines()[0] and len(cands.splitlines()) >= 2
    # yesterday: top pick appended to the CLV log with the last pregame price
    log = store["picks/clv_log.csv"].splitlines()
    assert len(log) == 2 and yday in log[1] and "EDM @ TBL O 5.5" in log[1]
    assert ",-120," in log[1]  # close price of the over
    # a second visit doesn't append the same day again
    at.run()
    assert len(store["picks/clv_log.csv"].splitlines()) == 2
