from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon=":bike:",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

COLOR_PRIMARY = "#2E86AB"
COLOR_SECONDARY = "#A23B72"
COLOR_ACCENT = "#F18F01"
COLOR_RED = "#C73E1D"
COLOR_GREEN = "#3A9D5D"
PALETTE = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_RED]

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
WEATHER_ORDER = ["Clear", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"]
DAY_TYPE_ORDER = ["Working Day", "Weekend/Holiday"]
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEMAND_TIER_ORDER = ["Low", "Medium", "High", "Very High"]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }
            .app-header {
                padding: 1.1rem 1.3rem;
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(46, 134, 171, 0.14), rgba(241, 143, 1, 0.16));
                border: 1px solid rgba(22, 32, 51, 0.08);
                margin-bottom: 1rem;
            }
            .app-header h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
                color: #152238;
            }
            .app-header p {
                margin: 0;
                color: #4d5c70;
                line-height: 1.55;
            }
            .section-note {
                padding: 0.8rem 1rem;
                border-left: 4px solid #2E86AB;
                background: rgba(46, 134, 171, 0.08);
                border-radius: 8px;
                margin: 0.5rem 0 1rem 0;
                color: #35445a;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid rgba(22, 32, 51, 0.08);
                border-radius: 10px;
                padding: 0.75rem 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = Path(__file__).resolve().parent
    day_path = base_path / "data" / "day.csv"
    hour_path = base_path / "data" / "hour.csv"

    day_df = pd.read_csv(day_path)
    hour_df = pd.read_csv(hour_path)

    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {
        1: "Clear",
        2: "Mist",
        3: "Light Rain/Snow",
        4: "Heavy Rain/Snow",
    }
    weekday_map = {
        0: "Sun",
        1: "Mon",
        2: "Tue",
        3: "Wed",
        4: "Thu",
        5: "Fri",
        6: "Sat",
    }
    year_map = {0: "2011", 1: "2012"}

    for dataset in (day_df, hour_df):
        dataset["dteday"] = pd.to_datetime(dataset["dteday"])
        dataset["season_label"] = pd.Categorical(
            dataset["season"].map(season_map),
            categories=SEASON_ORDER,
            ordered=True,
        )
        dataset["weather_label"] = pd.Categorical(
            dataset["weathersit"].map(weather_map),
            categories=WEATHER_ORDER,
            ordered=True,
        )
        dataset["weekday_label"] = pd.Categorical(
            dataset["weekday"].map(weekday_map),
            categories=WEEKDAY_ORDER,
            ordered=True,
        )
        dataset["year_label"] = dataset["yr"].map(year_map)
        dataset["month_name"] = pd.Categorical(
            dataset["dteday"].dt.strftime("%b"),
            categories=MONTH_ORDER,
            ordered=True,
        )
        dataset["day_type"] = np.where(dataset["workingday"].eq(1), "Working Day", "Weekend/Holiday")
        dataset["temp_celsius"] = (dataset["temp"] * 41).round(2)
        dataset["atemp_celsius"] = (dataset["atemp"] * 50).round(2)
        dataset["humidity_pct"] = (dataset["hum"] * 100).round(2)
        dataset["windspeed_kmh"] = (dataset["windspeed"] * 67).round(2)

    day_df["demand_tier"] = pd.qcut(
        day_df["cnt"],
        q=4,
        labels=DEMAND_TIER_ORDER,
    )
    hour_df["demand_period"] = pd.cut(
        hour_df["hr"],
        bins=[-1, 5, 11, 15, 19, 23],
        labels=["Late Night", "Morning Commute", "Midday", "Evening Commute", "Night"],
    )

    return day_df, hour_df


def filter_data(
    day_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp],
    years: list[str],
    seasons: list[str],
    weather: list[str],
    day_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    day_mask = (
        day_df["dteday"].between(start_date, end_date)
        & day_df["year_label"].isin(years)
        & day_df["season_label"].isin(seasons)
        & day_df["weather_label"].isin(weather)
        & day_df["day_type"].isin(day_types)
    )
    hour_mask = (
        hour_df["dteday"].between(start_date, end_date)
        & hour_df["year_label"].isin(years)
        & hour_df["season_label"].isin(seasons)
        & hour_df["weather_label"].isin(weather)
        & hour_df["day_type"].isin(day_types)
    )

    return day_df.loc[day_mask].copy(), hour_df.loc[hour_mask].copy()


def format_number(value: float) -> str:
    return f"{int(round(value)):,}"


def make_monthly_trend_chart(day_df: pd.DataFrame) -> alt.Chart:
    monthly_trend = (
        day_df.groupby(["year_label", "month_name"], observed=False)["cnt"]
        .mean()
        .dropna()
        .reset_index(name="average_rentals")
        .sort_values(["year_label", "month_name"])
    )

    return (
        alt.Chart(monthly_trend)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("month_name:N", sort=MONTH_ORDER, title="Bulan"),
            y=alt.Y("average_rentals:Q", title="Rata-Rata Peminjaman Harian"),
            color=alt.Color(
                "year_label:N",
                title="Tahun",
                scale=alt.Scale(range=[COLOR_PRIMARY, COLOR_ACCENT]),
            ),
            tooltip=[
                alt.Tooltip("year_label:N", title="Tahun"),
                alt.Tooltip("month_name:N", title="Bulan"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Rata-Rata Peminjaman Harian per Bulan", height=360)
    )


def make_season_chart(day_df: pd.DataFrame) -> alt.Chart:
    seasonal_avg = (
        day_df.groupby("season_label", observed=False)["cnt"]
        .mean()
        .dropna()
        .reset_index(name="average_rentals")
    )

    return (
        alt.Chart(seasonal_avg)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            y=alt.Y("season_label:N", sort=SEASON_ORDER, title="Musim"),
            x=alt.X("average_rentals:Q", title="Rata-Rata Peminjaman"),
            color=alt.Color(
                "season_label:N",
                scale=alt.Scale(domain=SEASON_ORDER, range=PALETTE),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("season_label:N", title="Musim"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Rata-Rata Peminjaman Harian Berdasarkan Musim", height=300)
    )


def make_weather_chart(day_df: pd.DataFrame) -> alt.Chart:
    weather_avg = (
        day_df.groupby("weather_label", observed=False)["cnt"]
        .mean()
        .dropna()
        .reset_index(name="average_rentals")
    )
    available_weather = [item for item in WEATHER_ORDER if item in set(weather_avg["weather_label"].astype(str))]

    return (
        alt.Chart(weather_avg)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("weather_label:N", sort=WEATHER_ORDER, title="Kondisi Cuaca"),
            y=alt.Y("average_rentals:Q", title="Rata-Rata Peminjaman"),
            color=alt.Color(
                "weather_label:N",
                scale=alt.Scale(domain=available_weather, range=PALETTE[: len(available_weather)]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("weather_label:N", title="Kondisi Cuaca"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Rata-Rata Peminjaman Harian Berdasarkan Kondisi Cuaca", height=300)
    )


def make_hourly_day_type_chart(hour_df: pd.DataFrame) -> alt.Chart:
    hourly_day_type = (
        hour_df.groupby(["hr", "day_type"])["cnt"]
        .mean()
        .reset_index(name="average_rentals")
    )

    return (
        alt.Chart(hourly_day_type)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("hr:O", title="Jam"),
            y=alt.Y("average_rentals:Q", title="Rata-Rata Peminjaman"),
            color=alt.Color(
                "day_type:N",
                title="Tipe Hari",
                sort=DAY_TYPE_ORDER,
                scale=alt.Scale(domain=DAY_TYPE_ORDER, range=[COLOR_PRIMARY, COLOR_SECONDARY]),
            ),
            tooltip=[
                alt.Tooltip("day_type:N", title="Tipe Hari"),
                alt.Tooltip("hr:O", title="Jam"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Pola Peminjaman per Jam pada Hari Kerja vs Akhir Pekan", height=360)
    )


def make_rider_profile_chart(hour_df: pd.DataFrame) -> alt.Chart:
    rider_profile = (
        hour_df.groupby("hr")[["casual", "registered"]]
        .mean()
        .reset_index()
        .melt(id_vars="hr", var_name="user_type", value_name="average_rentals")
    )

    return (
        alt.Chart(rider_profile)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("hr:O", title="Jam"),
            y=alt.Y("average_rentals:Q", title="Rata-Rata Peminjaman"),
            color=alt.Color(
                "user_type:N",
                title="Jenis Pengguna",
                scale=alt.Scale(domain=["casual", "registered"], range=[COLOR_ACCENT, COLOR_GREEN]),
            ),
            tooltip=[
                alt.Tooltip("user_type:N", title="Jenis Pengguna"),
                alt.Tooltip("hr:O", title="Jam"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Pola Pengguna Casual dan Registered per Jam", height=360)
    )


def make_weekday_hour_heatmap(hour_df: pd.DataFrame) -> alt.Chart:
    heatmap_data = (
        hour_df.groupby(["weekday_label", "hr"], observed=False)["cnt"]
        .mean()
        .dropna()
        .reset_index(name="average_rentals")
    )

    return (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X("hr:O", title="Jam"),
            y=alt.Y("weekday_label:N", sort=WEEKDAY_ORDER, title="Hari"),
            color=alt.Color("average_rentals:Q", title="Rata-Rata", scale=alt.Scale(scheme="tealblues")),
            tooltip=[
                alt.Tooltip("weekday_label:N", title="Hari"),
                alt.Tooltip("hr:O", title="Jam"),
                alt.Tooltip("average_rentals:Q", title="Rata-Rata", format=",.0f"),
            ],
        )
        .properties(title="Heatmap Rata-Rata Peminjaman Berdasarkan Hari dan Jam", height=330)
    )


def make_correlation_heatmap(day_df: pd.DataFrame) -> alt.Chart:
    correlation_columns = [
        "temp_celsius",
        "atemp_celsius",
        "humidity_pct",
        "windspeed_kmh",
        "casual",
        "registered",
        "cnt",
    ]
    corr = day_df[correlation_columns].corr().round(3)
    corr_long = corr.reset_index().melt(id_vars="index", var_name="variable", value_name="correlation")
    corr_long = corr_long.rename(columns={"index": "feature"})

    return (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("variable:N", sort=correlation_columns, title="Variabel"),
            y=alt.Y("feature:N", sort=correlation_columns, title="Variabel"),
            color=alt.Color(
                "correlation:Q",
                title="Korelasi",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Variabel 1"),
                alt.Tooltip("variable:N", title="Variabel 2"),
                alt.Tooltip("correlation:Q", title="Korelasi", format=".3f"),
            ],
        )
        .properties(title="Heatmap Korelasi Variabel Numerik", height=430)
    )


def make_demand_tier_chart(day_df: pd.DataFrame) -> alt.Chart:
    tier_share = pd.crosstab(
        day_df["season_label"],
        day_df["demand_tier"],
        normalize="index",
    )
    tier_long = (tier_share * 100).reset_index().melt(
        id_vars="season_label",
        var_name="demand_tier",
        value_name="share_pct",
    )

    return (
        alt.Chart(tier_long)
        .mark_rect()
        .encode(
            x=alt.X("demand_tier:N", sort=DEMAND_TIER_ORDER, title="Tier Permintaan"),
            y=alt.Y("season_label:N", sort=SEASON_ORDER, title="Musim"),
            color=alt.Color("share_pct:Q", title="Persentase", scale=alt.Scale(scheme="oranges")),
            tooltip=[
                alt.Tooltip("season_label:N", title="Musim"),
                alt.Tooltip("demand_tier:N", title="Tier"),
                alt.Tooltip("share_pct:Q", title="Persentase", format=".1f"),
            ],
        )
        .properties(title="Persentase Tier Permintaan Harian pada Setiap Musim", height=280)
    )


def show_metric_cards(day_df: pd.DataFrame, hour_df: pd.DataFrame) -> None:
    total_rentals = day_df["cnt"].sum()
    average_daily = day_df["cnt"].mean()
    registered_share = day_df["registered"].sum() / total_rentals * 100
    busiest_hour = hour_df.groupby("hr")["cnt"].mean().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Peminjaman", format_number(total_rentals))
    col2.metric("Rata-Rata Harian", format_number(average_daily))
    col3.metric("Share Registered", f"{registered_share:.1f}%")
    col4.metric("Jam Tersibuk", f"{int(busiest_hour):02d}:00")


def show_summary_tables(day_df: pd.DataFrame, hour_df: pd.DataFrame) -> None:
    top_days = (
        day_df.sort_values("cnt", ascending=False)
        .loc[:, ["dteday", "cnt", "season_label", "weather_label", "day_type"]]
        .head(10)
        .rename(
            columns={
                "dteday": "Tanggal",
                "cnt": "Total Peminjaman",
                "season_label": "Musim",
                "weather_label": "Cuaca",
                "day_type": "Tipe Hari",
            }
        )
    )
    top_days["Tanggal"] = top_days["Tanggal"].dt.strftime("%Y-%m-%d")

    peak_hours = (
        hour_df.groupby(["day_type", "hr"])["cnt"]
        .mean()
        .reset_index(name="Rata-Rata Peminjaman")
        .sort_values("Rata-Rata Peminjaman", ascending=False)
        .groupby("day_type")
        .head(3)
        .rename(columns={"day_type": "Tipe Hari", "hr": "Jam"})
    )
    peak_hours["Jam"] = peak_hours["Jam"].map(lambda value: f"{int(value):02d}:00")
    peak_hours["Rata-Rata Peminjaman"] = peak_hours["Rata-Rata Peminjaman"].round(1)

    table_col1, table_col2 = st.columns([1.1, 0.9])
    with table_col1:
        st.subheader("Top 10 Hari dengan Peminjaman Tertinggi")
        st.dataframe(top_days, use_container_width=True, hide_index=True)
    with table_col2:
        st.subheader("Top 3 Jam Puncak per Tipe Hari")
        st.dataframe(peak_hours, use_container_width=True, hide_index=True)


def main() -> None:
    apply_styles()
    day_df, hour_df = load_and_prepare_data()

    st.markdown(
        """
        <div class="app-header">
            <h1>Bike Sharing Dashboard</h1>
            <p>
                Dashboard interaktif untuk menganalisis pola peminjaman sepeda berdasarkan bulan, musim,
                cuaca, tipe hari, jam operasional, dan jenis pengguna.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Filter Data")
    min_date = day_df["dteday"].min().date()
    max_date = day_df["dteday"].max().date()
    selected_date = st.sidebar.date_input(
        "Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(selected_date) != 2:
        st.warning("Pilih tanggal awal dan akhir terlebih dahulu.")
        return

    selected_years = st.sidebar.multiselect(
        "Tahun",
        options=sorted(day_df["year_label"].unique()),
        default=sorted(day_df["year_label"].unique()),
    )
    selected_seasons = st.sidebar.multiselect(
        "Musim",
        options=SEASON_ORDER,
        default=SEASON_ORDER,
    )
    selected_weather = st.sidebar.multiselect(
        "Kondisi Cuaca",
        options=[item for item in WEATHER_ORDER if item in set(day_df["weather_label"].astype(str))],
        default=[item for item in WEATHER_ORDER if item in set(day_df["weather_label"].astype(str))],
    )
    selected_day_types = st.sidebar.multiselect(
        "Tipe Hari",
        options=DAY_TYPE_ORDER,
        default=DAY_TYPE_ORDER,
    )

    filtered_day, filtered_hour = filter_data(
        day_df,
        hour_df,
        selected_date,
        selected_years,
        selected_seasons,
        selected_weather,
        selected_day_types,
    )

    if filtered_day.empty or filtered_hour.empty:
        st.warning("Tidak ada data yang sesuai dengan filter saat ini. Silakan ubah kombinasi filter.")
        return

    show_metric_cards(filtered_day, filtered_hour)
    st.markdown(
        f"""
        <div class="section-note">
            Data aktif mencakup <strong>{len(filtered_day):,}</strong> observasi harian dan
            <strong>{len(filtered_hour):,}</strong> observasi per jam. Gunakan filter untuk melihat perubahan pola demand.
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, question_1_tab, question_2_tab, supporting_tab = st.tabs(
        ["Ringkasan", "Pertanyaan 1", "Pertanyaan 2", "Analisis Pendukung"]
    )

    with overview_tab:
        st.altair_chart(make_monthly_trend_chart(filtered_day), use_container_width=True)
        overview_col1, overview_col2 = st.columns(2)
        with overview_col1:
            st.altair_chart(make_season_chart(filtered_day), use_container_width=True)
        with overview_col2:
            st.altair_chart(make_weather_chart(filtered_day), use_container_width=True)
        show_summary_tables(filtered_day, filtered_hour)

    with question_1_tab:
        st.markdown(
            """
            <div class="section-note">
                Pertanyaan 1 berfokus pada perubahan rata-rata peminjaman sepeda per bulan,
                serta perbedaan demand menurut musim dan kondisi cuaca.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(make_monthly_trend_chart(filtered_day), use_container_width=True)
        q1_col1, q1_col2 = st.columns(2)
        with q1_col1:
            st.altair_chart(make_season_chart(filtered_day), use_container_width=True)
        with q1_col2:
            st.altair_chart(make_weather_chart(filtered_day), use_container_width=True)
        st.info(
            "Prioritas kapasitas operasional sebaiknya diarahkan pada bulan dan musim dengan "
            "rata-rata peminjaman tertinggi, terutama ketika cuaca cerah."
        )

    with question_2_tab:
        st.markdown(
            """
            <div class="section-note">
                Pertanyaan 2 membandingkan jam puncak hari kerja dan akhir pekan,
                lalu melihat pola pengguna casual dan registered.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(make_hourly_day_type_chart(filtered_hour), use_container_width=True)
        q2_col1, q2_col2 = st.columns(2)
        with q2_col1:
            st.altair_chart(make_rider_profile_chart(filtered_hour), use_container_width=True)
        with q2_col2:
            st.altair_chart(make_weekday_hour_heatmap(filtered_hour), use_container_width=True)
        st.info(
            "Hari kerja cenderung membentuk pola komuter pada pagi dan sore, sedangkan akhir pekan "
            "lebih kuat pada siang hingga sore."
        )

    with supporting_tab:
        support_col1, support_col2 = st.columns([1.05, 0.95])
        with support_col1:
            st.altair_chart(make_correlation_heatmap(filtered_day), use_container_width=True)
        with support_col2:
            st.altair_chart(make_demand_tier_chart(filtered_day), use_container_width=True)
        show_summary_tables(filtered_day, filtered_hour)


if __name__ == "__main__":
    main()
