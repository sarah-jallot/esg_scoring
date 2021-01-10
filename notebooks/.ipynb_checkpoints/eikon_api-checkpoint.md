### Querying the Refinitiv API


In this notebook we define a pythonic way to query the Refinitiv API for ESG Data and automate the process.

```python
#!pip install eikon
```

```python
!pip list
```

```python
import eikon as ek
```

```python
app_key = "467e1d8d1e624dfc834b60e91bba50e474bea063"
ek.set_app_key(app_key)
```

```python
ek.get_news_headlines('R:LHAG.DE', date_from='2019-03-06T09:00:00', date_to='2019-03-06T18:00:00')
```

```python
headlines = ek.get_news_headlines('EU AND POL',1)
story_id = headlines.iat[0,2]
ek.get_news_story(story_id)
```

```python
df = ek.get_timeseries(["MSFT.O"], 
                       start_date="2016-01-01",  
                       end_date="2016-01-10")
df
```

```python
df, err = ek.get_data(['GOOG.O','MSFT.O', 'FB.O'], 
                      [ 'TR.Revenue','TR.GrossProfit'])
df
```

```python
df, err = ek.get_data(['GOOG.O', 'MSFT.O', 'FB.O', 'AMZN.O', 'TWTR.K'], 
                      ['TR.Revenue.date','TR.Revenue','TR.GrossProfit'],
                      {'Scale': 6, 'SDate': 0, 'EDate': -2, 'FRQ': 'FY', 'Curn': 'EUR'})
df
```

```python
df, err = ek.get_data(['VOD.L', 'FB.O'], 
                      [ 'TR.Revenue', 'TR.GrossProfit', 'CF_LAST'])
df
```
