"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from price_elasticity import make_price_elasticity

st.set_page_config(layout="wide")


def _load_data() -> pd.DataFrame:
    df_raw = pd.read_csv("data/01_raw/df_ready.csv")
    return df_raw


def _drop_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "Unnamed: 0",
        "Cluster",
        "condition",
        "Disc_percentage",
        "isSale",
        "Imp_count",
        "p_description",
        "currency",
        "dateAdded",
        "dateSeen",
        "dateUpdated",
        "imageURLs",
        "shipping",
        "sourceURLs",
        "weight",
        "Date_imp_d.1",
        "Zscore_1",
        "price_std",
    ]

    df_raw = df_raw.drop(columns=columns_to_drop)
    df_raw.columns = [
        "date_imp",
        "date_imp_d",
        "category_name",
        "name",
        "price",
        "disc_price",
        "merchant",
        "brand",
        "manufacturer",
        "day_n",
        "month",
        "month_n",
        "day",
        "week_number",
    ]
    return df_raw


def _change_dtypes(df1: pd.DataFrame) -> pd.DataFrame:
    df1["date_imp_d"] = pd.to_datetime(df1["date_imp_d"])
    return df1


def _prepare_data(df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_best = df2.loc[
        (df2["category_name"] == "laptop, computer")
        & (df2["merchant"] == "Bestbuy.com")
    ]
    df_agg = (
        df_best.groupby(["name", "week_number"])
        .agg({"disc_price": "mean", "date_imp": "count"})
        .reset_index()
    )

    x_price = df_agg.pivot(index="week_number", columns="name", values="disc_price")

    y_demand = df_agg.pivot(index="week_number", columns="name", values="date_imp")

    median_price = np.round(x_price.median(), 2)
    x_price.fillna(median_price, inplace=True)
    y_demand.fillna(0, inplace=True)
    return x_price, y_demand


def _calculate_price_elasticity(
    x_price: pd.DataFrame, y_demand: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate price elasticity for each product.

    Parameters
    ----------
    x_price : pd.DataFrame
        The DataFrame of product prices.
    y_demand : pd.DataFrame
        The DataFrame of product demands.

    Returns
    -------
    pd.DataFrame
        The DataFrame with price elasticity results for each product.
    """
    price_elasticity_laptop = {
        "name": [],
        "price_elasticity": [],
        "price_mean": [],
        "quantity_mean": [],
        "intercept": [],
        "slope": [],
        "rsquared": [],
        "p_value": [],
    }

    for column in x_price.columns:
        column_points = [
            {
                "x_price": x_price.reset_index(drop=True)[column][i],
                "y_demand": y_demand.reset_index(drop=True)[column][i],
            }
            for i in range(len(x_price[column]))
        ]

        list_price = [point["x_price"] for point in column_points]
        list_demand = [point["y_demand"] for point in column_points]

        X = sm.add_constant(list_price)
        model = sm.OLS(list_demand, X)
        results = model.fit()

        if results.f_pvalue < 0.05:
            intercept, slope = results.params

            price_elasticity_laptop["name"].append(column)
            price_elasticity_laptop["price_elasticity"].append(
                slope * (np.mean(list_price) / np.mean(list_demand))
            )
            price_elasticity_laptop["rsquared"].append(results.rsquared)
            price_elasticity_laptop["p_value"].append(results.f_pvalue)
            price_elasticity_laptop["intercept"].append(intercept)
            price_elasticity_laptop["slope"].append(slope)
            price_elasticity_laptop["price_mean"].append(
                np.round(np.mean(list_price), 2)
            )
            price_elasticity_laptop["quantity_mean"].append(
                np.round(np.mean(list_demand), 2)
            )

    return pd.DataFrame(price_elasticity_laptop)


def simulate_elasticity(
    percentual: float, y_demand: pd.DataFrame, df_elasticity: pd.DataFrame, option: str
) -> Union[pd.DataFrame, None]:
    """
    Simulate price elasticity with a given percentage change.

    Parameters
    ----------
    percentual : float
        The percentage of price change.
    y_demand : pd.DataFrame
        The DataFrame of product demands.
    df_elasticity : pd.DataFrame
        The DataFrame of price elasticity.
    option : str
        The option of price change, can be either "Desconto" or "Aumento de Preço".

    Returns
    -------
    Union[pd.DataFrame, None]
        The DataFrame of simulated results. Returns None if the percentual is 0.
    """
    if percentual == 0:
        simulate_result = None
    else:
        result_revenue = {
            "name": [],
            "faturamento_atual": [],
            "faturamento_novo": [],
            "variacao_faturamento": [],
            "variacao_percentual": [],
        }

        if option == "Desconto":
            percentual = -percentual

        for i in range(len(df_elasticity)):
            current_price_mean = df_elasticity["price_mean"][i]
            current_demand = y_demand[df_elasticity["name"][i]].sum()

            if percentual < 0:
                price_change = current_price_mean * (1 - ((percentual * (-1)) / 100))
            else:
                price_change = (current_price_mean * percentual) + current_price_mean

            demand_increase = (percentual / 100) * df_elasticity["price_elasticity"][i]
            new_demand = current_demand + (current_demand * demand_increase)

            current_revenue = round(current_price_mean * current_demand, 2)
            new_revenue = round(price_change * new_demand, 2)

            revenue_variation = round(new_revenue - current_revenue, 2)
            percentage_variation = round(
                (new_revenue - current_revenue) / current_revenue, 2
            )

            result_revenue["name"].append(df_elasticity["name"][i])
            result_revenue["faturamento_atual"].append(current_revenue)
            result_revenue["faturamento_novo"].append(new_revenue)
            result_revenue["variacao_faturamento"].append(revenue_variation)
            result_revenue["variacao_percentual"].append(percentage_variation)

        simulate_result = pd.DataFrame(result_revenue)

    return simulate_result


def _generate_product_report(
    final: pd.DataFrame, op: str, number: int, report_intro: str
) -> str:
    products = []

    for i in range(len(final)):
        product = final["name"][i]
        current_revenue = final["faturamento_atual"][i]
        new_revenue = final["faturamento_novo"][i]
        action = "Increase" if op == "Aumento de Preço" else "Decrease"
        action2 = "Increase" if new_revenue > current_revenue else "Decrease"

        limited_product = product if len(product) <= 50 else product[:50]

        products.append(limited_product)
        product_report = (
            f"- {action} {number}% on product {limited_product}:"
            f" {action2} of revenue in R${abs(new_revenue)}\n"
        )
        report_intro += product_report
    total_products_analyzed = len(products)

    return total_products_analyzed, report_intro


def _generate_general_report(
    final: pd.DataFrame, total_products_analyzed: int, number: int
) -> str:
    total_new_revenue = final["faturamento_novo"].sum()
    total_current_revenue = final["faturamento_atual"].sum()
    revenue_variation = total_new_revenue - total_current_revenue

    impact = "INCREASES" if revenue_variation > 0 else "DECREASES"
    general_report = "\n## **Impact on revenue and demand in business as a whole:**\n"
    general_report += f"- Total of analyzed products: {total_products_analyzed}\n"
    general_report += (
        f"- With a discount of {number}% the revenue of your business "
        f"{impact}, being able to make the potential revenue of "
        f"your business can reach {round(total_new_revenue,2)}. "
        f"This represents a value of {round(abs(revenue_variation),2)}"
        f" {'more' if impact == 'INCREASES' else 'less'} than you "
        f"currently invoice.\n"
    )
    general_report += (
        f"- Percentage variation in revenue: {final['variacao_percentual'].sum()}%\n"
    )
    return general_report


def make_simulation_report(final: pd.DataFrame, op: str, number: int) -> str:
    """
    Generate a simulation report.

    Parameters
    ----------
    final : pd.DataFrame
        The DataFrame of simulation results.
    op : str
        The option of price change.
    number : int
        The percentage of price change.

    Returns
    -------
    str
        The simulation report.
    """
    report_intro = (
        "### **Our Artificial Intelligence model has generated a "
        "custom report simulating the effects that this price change "
        "can cause in Demand and Revenue:**\n\n"
    )
    total_products_analyzed, report_intro = _generate_product_report(
        final, op, number, report_intro
    )

    general_report = _generate_general_report(final, total_products_analyzed, number)

    final_report = report_intro + general_report
    return final_report


def prepare_data_and_calculate_elasticity() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = _load_data()
    df1 = _drop_columns(df_raw)

    df2 = _change_dtypes(df1)

    x_price, y_demand = _prepare_data(df2)

    df_elasticity = _calculate_price_elasticity(x_price, y_demand)

    return df_elasticity, y_demand


def run_simulation_tab(df_elasticity: pd.DataFrame, y_demand: pd.DataFrame) -> None:
    """
    Run simulation and display results on Streamlit.

    Parameters
    ----------
    df_elasticity : pd.DataFrame
        The DataFrame of price elasticity.
    y_demand : pd.DataFrame
        The DataFrame of product demands.
    """
    col1, col2 = st.columns((1, 1))
    with col1:
        st.markdown(
            (
                "<h2 style='text-align: center;'>Would you like to apply a "
                "discount or a price increase to the products?</h2>"
            ),
            unsafe_allow_html=True,
        )
        option = st.selectbox("", ("Price Increase", "Apply Discount"))
        if option == "Price Increase":
            op = "Price Increase"
        else:
            op = "Discount"

    with col2:
        st.markdown(
            '<h2 style="text-align: center;">What percentage of '
            + op
            + " would you like to apply?</h2>",
            unsafe_allow_html=True,
        )
        number = st.number_input("")

    if number != 0:
        final = simulate_elasticity(number, y_demand, df_elasticity, op)
        final2 = final.copy()
        final2.columns = [
            "Product",
            "Current Revenue",
            "Predicted Revenue with AI",
            "Revenue Variation",
            "Percentage Variation",
        ]
        st.dataframe(final2, use_container_width=True)

        report = make_simulation_report(final, op, number)
        st.markdown(report)


if __name__ == "__main__":
    st.header("Price Elasticity Project")

    df_elasticity, y_demand = prepare_data_and_calculate_elasticity()

    tab1, tab2 = st.tabs(["Price Elasticity", "Simulate scenarios"])

    with tab1:
        make_price_elasticity()
    with tab2:
        run_simulation_tab(df_elasticity, y_demand)
