import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import risktools as rt
import numpy as np
from st_aggrid import AgGrid
from functions import MV


st.title("Mulit-variate Portfolio Analysis app")


df = pd.read_pickle("data/prices.pki")

PO = MV.gen_payoffs(df, MV.payoffs_cx)

st.header("Historical Payoffs (Cx)")
st.line_chart(PO)

with st.sidebar:
    with st.form("MV_form"):
        dt = st.selectbox("Time-steps", list(MV.time_step.keys()), index=3)
        T = st.slider("Years", min_value=1, max_value=10)
        sims = st.number_input(
            label="Sims", min_value=500, max_value=5000, value=1000, step=500
        )
        npsims = st.number_input(
            label="Random Portfolios",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500,
        )
        submitted = st.form_submit_button("Update Params")
    st.write(
        """NTD - this was built Dec 2022 with an older version of streamlit (before editable dataframes were added)
            """
    )


params = MV.get_params(PO, dt)[0]
cor = MV.get_params(PO, dt)[1]

st.title("Historical Financial Parameters")
reload_data = False
if st.button("Reset data"):
    params = rt.fitOU_MV(PO, MV.time_step[dt], log_price=False)
    reload_data = True

params = AgGrid(
    params.reset_index(), reload_data=reload_data, editable=True, update_mode="MANUAL"
)["data"].set_index("index")
reload_data = False


out = MV.gen_mv(cor=cor, T=T, dt=dt, sims=sims, PO=PO, params=params)

if "x" not in st.session_state:
    st.session_state.x = 0
st.title("Simulated Future Price Paths")
if st.button("Next sim"):
    st.session_state.x += 1

st.line_chart(
    MV.dt_index(
        pd.DataFrame(out[:, st.session_state.x, :]),
        T=T,
        dt=MV.time_step[dt],
        freq=MV.r_freq[dt],
    )
)


payoffs = MV.gen_payoffs_charts(out=out, payoff_funcs=[MV.payoff_fixed] * 3)

st.write(
    """
    #### Asset payoff profile logic
    ```python
         def payoff_fixed(x):
        return x.sum(axis=0)
    ```
    Payoff function can be defined per asset
    """
)
st.title("Asset Payoff Distributions")
st.plotly_chart(payoffs[1], use_container_width=True)

st.plotly_chart(payoffs[0], use_container_width=True)

wts = MV.gen_wts(PO, npsims)
e_PO = MV.gen_PO_ef(out, payoff_funcs=[MV.payoff_fixed] * 3)
ef = MV.gen_ef(e_PO, wts)

ef_df = MV.gen_ef_table(ef, wts)

st.title("Gen Future Payoffs Boxplot")
with st.form("Boxplot"):
    resample = st.selectbox("Resample", list(MV.r_freq.keys()), index=1)
    submitted = st.form_submit_button("Gen boxplot")

if submitted:
    st.plotly_chart(
        MV.boxplot(
            out,
            T=T,
            dt=MV.time_step[dt],
            freq_mean=MV.r_freq[resample],
            freq_dt=MV.r_freq[dt],
        ),
        use_container_width=True,
    )

st.title("Efficient Frontier")
col1, col2 = st.columns([2, 6])

with col1:
    with st.form("Portfolio Plot"):
        p1 = st.number_input("p1", min_value=0.0, max_value=1.0, step=0.01, value=0.33)
        p2 = st.number_input("p2", min_value=0.0, max_value=1.0, step=0.01, value=0.33)
        p3 = st.number_input("p3", min_value=0.0, max_value=1.0, step=0.01, value=0.34)
        st.write("Must sum to one:", sum([p1, p2, p3]))
        p_submit = st.form_submit_button("Plot Portfolio")

with col2:
    st.plotly_chart(
        MV.plot_pt(e_PO, [p1, p2, p3], MV.gen_ef_plot(ef_df)), use_container_width=True
    )
