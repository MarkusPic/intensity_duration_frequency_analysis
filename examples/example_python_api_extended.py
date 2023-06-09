from pathlib import Path

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd

# sub-folder for the results

output_directory = Path(__file__).parent / 'ehyd_112086_2019_idf_data'
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
series = pd.read_parquet('ehyd_112086.parquet')['N-Minutensummen-112086']

# setting the series for the analysis
idf.set_series(series)

# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(output_directory / 'idf_parameters_new.yaml')
exit()
# --------
# idf.write_return_periods_frame(path.join(output_directory, 'idf_return_periods_frame.parq'))
# exit()
# idf.auto_save_return_periods_frame(path.join(output_directory, 'idf_return_periods_frame.parq'))

# --------
# events = idf.rain_events
# idf.add_max_return_periods_to_events(events)
# idf.write_rain_events(path.join(output_directory, 'events.csv'), sep=',', decimal='.')
# exit()
# idf.auto_save_rain_events(path.join(output_directory, 'events.csv'), sep=',', decimal='.')

# --------
# idf.event_report(path.join(output_directory, 'idf_event_analysis.pdf'), min_event_rain_sum=60, min_return_period=0.5, durations=None)

# --------
# idf.event_return_period_report(path.join(output_directory, 'idf_event_return_period_analysis.pdf'))

# --------
e = idf.rain_events

e = e[e[COL.LP] > 10]
# idf.add_max_return_periods_to_events(idf.rain_events)
# e = e[e[COL.MAX_PERIOD] > 2]

event = idf.rain_events.loc[125]


fig, caption = idf.event_plot(event, durations=idf.duration_steps[:11])
fig.tight_layout()
fig.show()

fig, caption = idf.event_plot(event)
fig.tight_layout()
fig.show()

rpf = idf.return_periods_frame[event[COL.START]:event[COL.END]]