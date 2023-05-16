from datetime import datetime
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt


with open('documentation.md', 'r') as f:
    DOCUMENTATION = f.read()

FILENAME = 'CSVFile.csv'
START_TIME_FIELD = ' Start time'
END_TIME_FIELD = ' End time'
TASK_NAME_FIELD = 'Task name'
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600


def get_average_day(df: pd.DataFrame):
    first_date = datetime.strptime(
        df[START_TIME_FIELD].iloc[0], '%b %d, %Y at %I:%M:%S %p')
    last_date = datetime.strptime(
        df[START_TIME_FIELD].iloc[-1], '%b %d, %Y at %I:%M:%S %p')
    time_delta = last_date - first_date
    days = round(time_delta.days + time_delta.seconds / SECONDS_PER_DAY, 1)
    grouped = df.groupby(TASK_NAME_FIELD)['Duration in hours'].agg(sum)

    fig, ax = plt.subplots(dpi=150)
    titles, vals = zip(*sorted(zip(grouped.index, grouped), key=lambda p: p[1]))
    plt.barh(titles, vals)
    plt.title(f'Daily activity means over {days} days')

    for bar, val in zip(ax.patches, vals):
        if round(val / days, 1) < 1:
            duration = f'{round(val / days * 60)}min'
        else:
            duration = f'{round(val / days, 1)}h'
        
        if val / sum(vals) <= .25:
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, 
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
    return get_average_day(df_trunc)


def get_moving_average(df: pd.DataFrame, window: int = 3):
    df[START_TIME_FIELD] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[START_TIME_FIELD]
    ])
    df[END_TIME_FIELD] = pd.to_datetime([
        datetime.strptime(x, '%b %d, %Y at %I:%M:%S %p') \
        for x in df[END_TIME_FIELD]
    ])

    df['Duration'] = df[END_TIME_FIELD] - df[START_TIME_FIELD]
    df['Duration'] = df['Duration'].dt.total_seconds() / SECONDS_PER_HOUR
    df['Date'] = df[START_TIME_FIELD].dt.date
    daily_duration = df.groupby(['Date', TASK_NAME_FIELD])['Duration'].sum().reset_index()
    daily_duration['Moving Average'] = daily_duration.groupby(TASK_NAME_FIELD)['Duration'].transform(lambda x: x.rolling(window=window).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    for task in daily_duration[TASK_NAME_FIELD].unique():
        task_data = daily_duration[daily_duration[TASK_NAME_FIELD] == task]
        ax.plot(task_data['Date'], task_data['Moving Average'], label=task)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Time spent per day (moving average)')
    ax.tick_params(axis='x', rotation=45)
    
    return fig


df = pd.read_csv(FILENAME)

df.drop([' Task description', ' Note', ' Tag'], axis=1, inplace=True)
first_date = datetime.strptime(
    df[START_TIME_FIELD].iloc[0], '%b %d, %Y at %I:%M:%S %p')
last_date = datetime.strptime(
    df[START_TIME_FIELD].iloc[-1], '%b %d, %Y at %I:%M:%S %p')

st.title('Activity analysis')
st.markdown(DOCUMENTATION)

st.header('Average day')
st.pyplot(get_average_day(df.copy()))

st.header('Average day in range')
start_date = datetime.combine(st.date_input('Start date', first_date.date()), datetime.min.time())
end_date = datetime.combine(st.date_input('End date', last_date.date()), datetime.max.time())
st.pyplot(get_average_day_in_range(df.copy(), start_date, end_date))

st.header('Moving average')
window = st.slider('Window size', 1, 7, 3)
st.pyplot(get_moving_average(df.copy(), window))
