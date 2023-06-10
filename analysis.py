from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.dates as mdates
from matplotlib import pyplot as plt


with open('documentation.md', 'r') as f:
    DOCUMENTATION = f.read()

ACTIVITY_FILENAME = 'CSVFile.csv'
STEPS_FILENAME = 'Pedometer.csv'
START_TIME_FIELD = ' Start time'
END_TIME_FIELD = ' End time'
TASK_NAME_FIELD = 'Task name'
DURATION_FIELD = 'Duration in hours'
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600


def set_ticks(x: pd.Series, ax: plt.Axes):
    """
    Set ticks for ax based on the date range x
    """
    print(x, x.max())
    date_range = x.max() - x.min()

    ax.tick_params(axis='x', rotation=45)
    if date_range < pd.Timedelta(weeks=1):
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    elif date_range < pd.Timedelta(weeks=12):
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))


def get_daily_chart(activity_df: pd.DataFrame, steps_df: pd.DataFrame, activities: list, raw: bool = False):
    fig, ax = plt.subplots(dpi=150)

    if len(activities) == 0:
        set_ticks(steps_df['Date'], ax)

        window_size = 1 if raw else 3
        steps_df['rolling_mean'] = steps_df['Steps'].rolling(window_size, min_periods=1).mean()
        ax.plot(steps_df['Date'], steps_df['rolling_mean'])
        ax.set_ylabel('Steps')
        ax.yaxis.set_major_formatter(lambda x, p: f'{x/1000:.1f}k')

        return fig

    activity_df[START_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in activity_df[START_TIME_FIELD]
    ])
    activity_df[END_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in activity_df[END_TIME_FIELD]
    ])
    activity_df['StartDate'] = activity_df[START_TIME_FIELD + '_tmp'].dt.date
    activity_df = activity_df[activity_df[TASK_NAME_FIELD].isin(activities)]

    steps_df = steps_df[steps_df['Date'] >= activity_df[START_TIME_FIELD + '_tmp'].min()]
    
    set_ticks(activity_df['StartDate'], ax)
    activity_df.groupby(['StartDate', TASK_NAME_FIELD])[DURATION_FIELD].sum().unstack().interpolate().plot.line(ax=ax, marker='.')
    ax.set_xlabel('Date')
    ax.set_ylabel('Duration (hours)')

    # Plot step data
    steps_ax = ax.twinx()
    set_ticks(activity_df['StartDate'], ax)
    steps_ax.plot(steps_df['Date'], steps_df['Steps'], '--', c='black', alpha=0.5)
    steps_ax.set_ylabel('Steps')
    steps_ax.yaxis.set_major_formatter(lambda x, p: f'{x/1000:.1f}k')
    
    # Clean up
    activity_df.drop([START_TIME_FIELD + '_tmp', END_TIME_FIELD + '_tmp', 'StartDate'], axis=1, inplace=True)

    return fig


def get_average_day(df: pd.DataFrame):
    first_date = datetime.strptime(
        df[START_TIME_FIELD].iloc[0], '%b %d, %Y at %I:%M:%S %p')
    last_date = datetime.strptime(
        df[START_TIME_FIELD].iloc[-1], '%b %d, %Y at %I:%M:%S %p')

    # Once again, fix ATracker's order
    if last_date < first_date:
        first_date, last_date = last_date, first_date

    time_delta = last_date - first_date
    days = round(time_delta.days + time_delta.seconds / SECONDS_PER_DAY, 1)
    grouped = df.groupby(TASK_NAME_FIELD)[DURATION_FIELD].agg(sum)

    fig, ax = plt.subplots(dpi=150)
    ax.set_xlabel('Total duration (hours)')
    titles, vals = zip(*sorted(zip(grouped.index, grouped), key=lambda p: p[1]))
    ax.barh(titles, vals)
    ax.set_title(f'Daily activity means over {days} days')

    for bar, val in zip(ax.patches, vals):
        if round(val / days, 1) < 1:
            duration = f'{round(val / days * 60)}min'
        else:
            duration = f'{round(val / days, 1)}h'
        
        if round(val / sum(vals) * 100, 1) <= 20:
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2, 
                    f'{duration} ({round(val/sum(vals)*100, 1)}%)', 
                    color='black', ha='left', va='center', fontdict={'size': 8})
        else:
            ax.text(val / 2, bar.get_y() + bar.get_height() / 2, 
                    f'{duration} ({round(val/sum(vals)*100, 1)}%)', 
                    color='white', ha='left', va='center', fontdict={'size': 8})
    
    return fig


def get_average_day_in_range(df: pd.DataFrame, start: datetime, end: datetime):
    df[START_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[START_TIME_FIELD]
    ])
    df[END_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[END_TIME_FIELD]
    ])

    df_trunc = df[(df[START_TIME_FIELD + '_tmp'] >= start) & (df[END_TIME_FIELD + '_tmp'] <= end)].reset_index()
    assert len(df_trunc) <= len(df)

    # Clean up
    df.drop([START_TIME_FIELD + '_tmp', END_TIME_FIELD + '_tmp'], axis=1, inplace=True)

    return get_average_day(df_trunc)


def get_moving_average(df: pd.DataFrame, window: int = 3):
    df[START_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[START_TIME_FIELD]
    ])
    df[END_TIME_FIELD + '_tmp'] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[END_TIME_FIELD]
    ])

    df['Duration'] = df[END_TIME_FIELD + '_tmp'] - df[START_TIME_FIELD + '_tmp']
    df['Duration'] = df['Duration'].dt.total_seconds() / SECONDS_PER_HOUR
    df['Date'] = df[START_TIME_FIELD + '_tmp'].dt.date
    daily_duration = df.groupby(['Date', TASK_NAME_FIELD])['Duration'].sum().reset_index()
    daily_duration['Moving Average'] = daily_duration.interpolate().groupby(TASK_NAME_FIELD)['Duration'].transform(lambda x: x.rolling(window=window).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    set_ticks(df['Date'], ax)
    for task in daily_duration[TASK_NAME_FIELD].unique():
        task_data = daily_duration[daily_duration[TASK_NAME_FIELD] == task]
        ax.plot(task_data['Date'], task_data['Moving Average'], label=task)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Time spent per day (moving average)')

    # Clean up
    df.drop([START_TIME_FIELD + '_tmp', END_TIME_FIELD + '_tmp', 'Duration', 'Date'], axis=1, inplace=True)
    
    return fig


activity_df = pd.read_csv(ACTIVITY_FILENAME)
activity_df[TASK_NAME_FIELD] = activity_df[TASK_NAME_FIELD].apply(lambda x: x.strip())
activity_df.drop([' Task description', ' Note', ' Tag'], axis=1, inplace=True)

steps_df = pd.read_csv(STEPS_FILENAME)
steps_df['Date'] = pd.to_datetime(steps_df['Date'])

first_date = datetime.strptime(
    activity_df[START_TIME_FIELD].iloc[0], '%b %d, %Y at %I:%M:%S %p')
last_date = datetime.strptime(
    activity_df[START_TIME_FIELD].iloc[-1], '%b %d, %Y at %I:%M:%S %p')

# ATracker can't seem to figure out which way to order things
if first_date > last_date:
    first_date, last_date = last_date, first_date

st.title('Activity analysis')
st.markdown(DOCUMENTATION)

st.header('Average day')
start_date = datetime.combine(st.date_input('Start date', first_date.date()), datetime.min.time())
end_date = datetime.combine(st.date_input('End date', last_date.date()), datetime.max.time())
st.pyplot(get_average_day_in_range(activity_df.copy(), start_date, end_date))

st.header('Daily activity')
activities = st.multiselect('Activities', activity_df[TASK_NAME_FIELD].unique())

if len(activities) == 0:
    st.markdown('Since the date range is large, the plot shows moving average with a window size of 3.')
    raw = st.checkbox('Show raw data instead')
else:
    raw = False

st.pyplot(get_daily_chart(activity_df.copy(), steps_df.copy(), activities, raw=raw))

st.header('Moving average')
window = st.slider('Window size', 1, 7, 3)
st.pyplot(get_moving_average(activity_df.copy(), window))
