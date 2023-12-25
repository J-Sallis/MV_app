import pandas as pd
import streamlit as st
import risktools as rt
from ridgeplot import ridgeplot
import plotly.express as px
from plotly.subplots import make_subplots


time_step = {"Yearly": 1, "Monthly": 1 / 12, "Weekly": 1 / 52, "Daily": 1 / 252}
r_freq = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W-Mon", "Daily": "B"}
payoffs_cx = {"ULSD": "ULSD - WTI", "Gas": "Gas - WTI", "Jet": "Jet - WTI"}


@st.cache()
def dt_index(df, T, dt, freq="B"):
    """Creates a df index of future dates based on forward time steps

    Args:
        df (pd.DataFrame): df of forward looking prices/payoffs
        T (int): # of years
        dt (float): time step in a year
        freq (str, optional): Pandas date freq. Defaults to "B".

    Returns:
        df: _description_
    """
    df.index = pd.date_range(
        start=pd.Timestamp.now(), periods=int(T / dt) + 1, freq=freq
    )
    return df


@st.cache(allow_output_mutation=True)
def gen_payoffs_charts(out, payoff_funcs):
    payoffs = rt.calculate_payoffs(out, payoff_funcs=payoff_funcs)
    r_fig = ridgeplot(samples=payoffs.T)
    fig = px.histogram(
        pd.DataFrame(payoffs).melt(),
        facet_col="variable",
        nbins=100,
    )
    return r_fig, fig


@st.cache()
def gen_PO_ef(out, payoff_funcs):
    payoffs = rt.calculate_payoffs(out, payoff_funcs=payoff_funcs)
    return payoffs


@st.cache()
def payoff_fixed(x):
    # ret = np.clip(x, 0, None)
    return x.sum(axis=0)


@st.cache()
def gen_mv(cor, T, dt, sims, PO, params):
    eps = rt.generate_eps_MV(cor=cor, T=T, dt=time_step[dt], sims=sims)
    out = rt.simOU_MV(
        s0=PO.iloc[-1],
        mu=params.loc["mu"],
        theta=params.loc["theta"],
        sigma=params.loc["annualized_sigma"],
        sims=sims,
        eps=eps,
        T=T,
        dt=dt,
    )
    return out


@st.cache()
def gen_payoffs(df, payoff_forms):
    PO = rt.calc_spread_MV(df, payoff_forms)
    return PO


@st.cache()
def get_params(PO, dt):
    params = rt.fitOU_MV(PO, time_step[dt], log_price=False)
    cor = PO.diff().corr()
    return params, cor


@st.cache()
def boxplot(
    array,
    freq_mean="MS",
    colors=["red", "blue", "orange"],
    assets=[0, 1, 2],
    T=5,
    dt=1 / 252,
    freq_dt="B",
):
    if len(colors) != array.shape[2]:
        raise ValueError("Provided colors does not match number of simulated assets")
    if len(assets) != array.shape[2]:
        raise ValueError("Asset list does not match number of simulated assets")
    box_figs = []
    for i in range(0, array.shape[2]):
        fig = px.box(
            dt_index(pd.DataFrame(array[:, :, i]), T=T, dt=dt, freq=freq_dt)
            .resample(freq_mean)
            .mean()
            .T
        )
        fig.update_traces(
            line=dict(color=colors[i]), marker_color=colors[i], name=assets[i]
        )
        box_figs.append(fig.data[0])
    box = make_subplots()
    box.add_traces(data=box_figs).update_traces(showlegend=True)
    return box


@st.cache()
def gen_wts(PO, nsims):
    wts = rt.generate_random_portfolio_weights(PO.shape[1], number_sims=nsims)
    return wts


@st.cache()
def gen_ef(payoffs, wts):
    ef = rt.simulate_efficient_frontier(payoffs, wts)
    return ef


@st.cache()
def gen_ef_table(ef, wts):
    df = rt.make_efficient_frontier_table(ef, wts)
    return df


# @st.cache(allow_output_mutation=True)
def gen_ef_plot(df):
    fig = rt.plot_efficient_frontier(df)
    return fig


@st.cache()
def plot_pt(PO, wt, ef_fig, **kwargs):
    fig = rt.plot_portfolio(PO, wt, ef_fig, **kwargs)
    return fig
