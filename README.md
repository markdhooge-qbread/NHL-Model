# NHL Model

NHL game flow and player prop model, synced from Google Colab.

## What's here

`notebooks/player_model.ipynb` has three cells:

1. **Team slate sim.** Runs 10,000 Gamma-Poisson Monte Carlo sims for each game on today's slate (America/New_York date). It uses MoneyPuck 5v5 and special-teams xG, B2B fatigue, and regressed goalie GSAx with a goalie swap widget.
2. **Player props.** Projects SOG, xG / anytime goal %, and points / 1+ point % for the top 8 skaters on each team in a chosen matchup.
3. **Box score backtest.** Compares model projections to the live or final NHL API box score for a chosen game.

Each cell is self-contained and pulls fresh data from MoneyPuck (`SEASON = 2026`) and `api-web.nhle.com`.

## Streamlit app

`app.py` is a Streamlit version of the notebook, with the same three views as tabs (Slate, Player Props, Live Tracker). It also has a date picker and a goalie override panel that applies to both the slate and props.

- Run locally: `pip install -r requirements.txt`, then `streamlit run app.py`
- Deploy: on Streamlit Community Cloud, create an app from this repo, choose branch `main`, and set the main file to `app.py`.

## Notebook

Open the notebook in Google Colab and run the cells, or run it locally in Jupyter after `pip install -r requirements.txt`.

If you change the model in the notebook, make the same change in `app.py`. The two are separate copies of the logic.

## Syncing from Colab

1. Open the notebook in Google Colab.
2. Go to **File > Save a copy in GitHub**.
3. Choose repository `markdhooge-qbread/NHL-Model`, branch `main`, and file path `notebooks/player_model.ipynb`.
4. Add a commit message and click **OK**.

Repeat these steps any time the notebook changes. You can also download the `.ipynb` and upload it to the repo on github.com.
